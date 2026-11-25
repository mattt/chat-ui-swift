import AnyLanguageModel
import Foundation
import SwiftData
import Synchronization
import Testing

private nonisolated struct ModelRequest: Sendable {
    let model: Model
    let session: LanguageModelSession
    let history: [String]
    let prompt: String
}

private nonisolated struct FixtureModel: LanguageModel {
    typealias UnavailableReason = Never
    let selection: Model
    var delayedReply: (@Sendable () async -> String)?
    let record: @Sendable (ModelRequest) -> Void

    func streamResponse<Content: Generable>(
        within session: LanguageModelSession, to prompt: Prompt, generating type: Content.Type,
        includeSchemaInPrompt: Bool, options: GenerationOptions
    ) -> sending LanguageModelSession.ResponseStream<Content> {
        let history = session.transcript.flatMap { entry -> [String] in
            let segments: [Transcript.Segment]
            let role: String
            switch entry {
            case .prompt(let turn): (role, segments) = ("user", turn.segments)
            case .response(let turn): (role, segments) = ("assistant", turn.segments)
            default: return []
            }
            return segments.compactMap {
                guard case .text(let text) = $0 else { return nil }
                return "\(role): \(text.content)"
            }
        }
        record(ModelRequest(model: selection, session: session, history: history, prompt: prompt.description))
        if delayedReply != nil {
            return .init(stream: AsyncThrowingStream { $0.finish() })
        }
        let reply = "Reply to \(prompt.description)"
        return .init(content: reply as! Content, rawContent: GeneratedContent(reply))
    }

    func respond<Content: Generable>(
        within session: LanguageModelSession, to prompt: Prompt, generating type: Content.Type,
        includeSchemaInPrompt: Bool, options: GenerationOptions
    ) async throws -> LanguageModelSession.Response<Content> {
        let respond = try #require(delayedReply)
        let reply = await respond()
        return .init(content: reply as! Content, rawContent: GeneratedContent(reply), transcriptEntries: [])
    }
}

@Suite(.timeLimit(.minutes(1)))
@MainActor
final class ChatModelSwitchingTests {
    private let container: ModelContainer
    private let requests = Mutex<[ModelRequest]>([])

    init() throws {
        container = try ModelContainer(
            for: Chat.self, Message.self, configurations: ModelConfiguration(isStoredInMemoryOnly: true)
        )
    }

    private func fixture(for model: Model) -> FixtureModel {
        FixtureModel(selection: model) { request in
            self.requests.withLock { $0.append(request) }
        }
    }

    @Test func switchingModelsContinuesTheSameChat() async throws {
        let viewModel = ChatViewModel(modelContext: container.mainContext) { self.fixture(for: $0) }
        let chat = viewModel.createNewChat()
        chat.title = "Fixture"
        let models: [Model] = [.system, .system, .huggingFace("one"), .huggingFace("two"), .system]

        for (index, model) in models.enumerated() {
            chat.updateModel(model)
            await viewModel.sendMessage(to: chat, content: String(index), using: chat.model)
        }

        let calls = requests.withLock { $0 }
        #expect(calls.map(\.model) == models)
        #expect(calls.map(\.prompt) == ["0", "1", "2", "3", "4"])
        try #require(calls.count == models.count)
        #expect(calls[0].session === calls[1].session)
        #expect(calls[0].session !== calls[4].session)
        for index in 2 ..< models.count {
            let priorTurns = (0 ..< index).flatMap { ["user: \($0)", "assistant: Reply to \($0)"] }
            #expect(calls[index].history == priorTurns + ["user: \(index)"])
        }
    }

    @Test func changingSelectionDuringASendAppliesToTheFollowingSend() async throws {
        let (started, signal) = AsyncStream<Void>.makeStream()
        let (resume, gate) = AsyncStream<Void>.makeStream()
        let viewModel = ChatViewModel(modelContext: container.mainContext) { model in
            if model == .system {
                signal.finish()
                for await _ in resume {}
            }
            return self.fixture(for: model)
        }
        let chat = viewModel.createNewChat()
        chat.title = "Fixture"
        let pending = Task { [model = chat.model] in
            await viewModel.sendMessage(to: chat, content: "first", using: model)
        }
        for await _ in started {}
        chat.updateModel(.huggingFace("next"))
        gate.finish()
        await pending.value
        await viewModel.sendMessage(to: chat, content: "second", using: chat.model)
        #expect(requests.withLock { $0.map(\.model) } == [.system, .huggingFace("next")])
    }

    @Test func stoppedReplyCannotFinishTheNextSend() async throws {
        let (replies, pending) = AsyncStream<CheckedContinuation<String, Never>>.makeStream()
        var iterator = replies.makeAsyncIterator()
        let viewModel = ChatViewModel(modelContext: container.mainContext) { model in
            var fixture = self.fixture(for: model)
            fixture.delayedReply = {
                await withCheckedContinuation { pending.yield($0) }
            }
            return fixture
        }
        let chat = viewModel.createNewChat()
        chat.title = "Fixture"

        let first = Task { [model = chat.model] in
            await viewModel.sendMessage(to: chat, content: "first", using: model)
        }
        let oldReply = try #require(await iterator.next())
        viewModel.stopGenerating()

        chat.updateModel(.huggingFace("next"))
        let second = Task { [model = chat.model] in
            await viewModel.sendMessage(to: chat, content: "second", using: model)
        }
        let newReply = try #require(await iterator.next())
        oldReply.resume(returning: "Late reply")
        await first.value
        #expect(viewModel.isGenerating)
        newReply.resume(returning: "Current reply")
        await second.value

        #expect(!viewModel.isGenerating)
        #expect(chat.messages.filter { !$0.isUser }.map(\.content) == ["Current reply"])
        #expect(requests.withLock { $0.map(\.model) } == [.system, .huggingFace("next")])
    }
}
