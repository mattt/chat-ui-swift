import Foundation
import SwiftData
import Observation
import AnyLanguageModel

@Observable
@MainActor
final class ChatViewModel {
    var isGenerating: Bool { currentTask != nil }
    private(set) var errorMessage: String?

    private let modelContext: ModelContext
    private var sessions: [UUID: (model: Model, session: LanguageModelSession)] = [:]
    private let makeLanguageModel: (Model) async throws -> any LanguageModel
    private var currentTask: Task<Void, Never>?

    convenience init(modelContext: ModelContext, authManager: AuthenticationManager? = nil) {
        self.init(modelContext: modelContext) { model in
            try await model.makeLanguageModel(authManager: authManager)
        }
    }

    init(modelContext: ModelContext, makeLanguageModel: @escaping (Model) async throws -> any LanguageModel) {
        self.modelContext = modelContext
        self.makeLanguageModel = makeLanguageModel
    }

    func stopGenerating() {
        currentTask?.cancel()
        currentTask = nil
    }

    func sendMessage(to chat: Chat, content: String, using model: Model) async {
        guard !content.isEmpty else { return }

        let userMessage = Message(content: content, isUser: true)
        userMessage.chat = chat
        chat.messages.append(userMessage)
        chat.updatedAt = Date()

        modelContext.insert(userMessage)

        do {
            try modelContext.save()
        } catch {
            errorMessage = "Failed to save message: \(error.localizedDescription)"
            return
        }

        let task = Task {
            await generateResponse(for: chat, to: userMessage, using: model)
        }
        currentTask = task
        await task.value
        if currentTask == task {
            currentTask = nil
        }
    }

    private func generateResponse(for chat: Chat, to message: Message, using model: Model) async {
        guard !Task.isCancelled else { return }
        errorMessage = nil
        let retained = sessions.removeValue(forKey: chat.id)
        var assistantMessage: Message?

        do {
            let languageModel = try await makeLanguageModel(model)
            try Task.checkCancellation()

            let session: LanguageModelSession
            if let retained, retained.model == model {
                session = retained.session
            } else {
                let history = chat.messages
                    .filter { $0.id != message.id && !$0.content.isEmpty }
                    .sorted { $0.timestamp < $1.timestamp }
                    .map { message -> Transcript.Entry in
                        let segments: [Transcript.Segment] = [.text(.init(content: message.content))]
                        if message.isUser {
                            return .prompt(.init(segments: segments))
                        }
                        return .response(.init(assetIDs: [], segments: segments))
                    }
                session = LanguageModelSession(model: languageModel, transcript: Transcript(entries: history))
            }
            let prompt = message.content

            // Create an assistant message immediately and update its content as we stream
            let reply = Message(content: "", isUser: false)
            reply.chat = chat
            chat.messages.append(reply)
            chat.updatedAt = Date()

            modelContext.insert(reply)
            assistantMessage = reply

            var receivedAnyStreamContent = false

            // Prefer streaming response when available
            let stream = session.streamResponse(to: prompt)
            var lastLength = 0
            for try await snapshot in stream {
                try Task.checkCancellation()
                receivedAnyStreamContent = true
                // Materialize a concrete String from the raw content.
                let full = (try? String(snapshot.rawContent)) ?? reply.content
                // Append only the new delta to avoid duplication when snapshots are cumulative.
                if full.count >= lastLength {
                    let startIdx = full.index(full.startIndex, offsetBy: lastLength)
                    let delta = String(full[startIdx...])
                    reply.content += delta
                    lastLength = full.count
                } else {
                    // If content shrank (rare), reset to full.
                    reply.content = full
                    lastLength = full.count
                }
                chat.updatedAt = Date()
            }
            try Task.checkCancellation()

            // Fallback to non-streaming if no streamed content arrived (e.g., model doesn't support streaming)
            if !receivedAnyStreamContent {
                let response = try await session.respond(to: prompt)
                try Task.checkCancellation()
                reply.content = response.content
                chat.updatedAt = Date()
            }

            try modelContext.save()
            if chat.modelContext != nil && !chat.isDeleted {
                sessions[chat.id] = (model, session)
            }

            if chat.title == nil {
                Task {
                    await generateTitle(for: chat)
                }
            }
        } catch where Task.isCancelled || error is CancellationError {
            if let reply = assistantMessage {
                if reply.content.isEmpty {
                    chat.messages.removeAll { $0.id == reply.id }
                    modelContext.delete(reply)
                }
                try? modelContext.save()
            }
        } catch {
            print("Error generating response: \(error)")
            errorMessage = "Failed to generate response: \(error.localizedDescription)"
        }
    }

    private func generateTitle(for chat: Chat) async {
        guard #available(macOS 26, iOS 26, visionOS 26, *) else { return }
        do {
            let systemModel = SystemLanguageModel()
            let session = LanguageModelSession(model: systemModel)

            let conversationText = chat.messages
                .map { "\($0.isUser ? "User" : "Assistant"): \($0.content)" }
                .joined(separator: "\n")

            let titlePrompt = """
                Generate a short, concise title (5 words or less) for this conversation:

                \(conversationText)

                Return only the title, nothing else.
                """

            let response = try await session.respond(to: titlePrompt)

            let title = response.content
                .trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))

            await MainActor.run {
                chat.title = title
                try? self.modelContext.save()
            }
        } catch {
            print("Failed to generate title: \(error.localizedDescription)")
            //            await MainActor.run {
            //                self.errorMessage = "Failed to generate title: \(error.localizedDescription)"
            //            }
        }
    }

    func createNewChat(model: Model = .system) -> Chat {
        let chat = Chat(model: model)
        modelContext.insert(chat)

        do {
            try modelContext.save()
        } catch {
            errorMessage = "Failed to create chat: \(error.localizedDescription)"
        }

        return chat
    }

    func deleteChat(_ chat: Chat) {
        sessions[chat.id] = nil
        modelContext.delete(chat)

        do {
            try modelContext.save()
        } catch {
            errorMessage = "Failed to delete chat: \(error.localizedDescription)"
        }
    }
}

extension Model {
    func makeLanguageModel(authManager: AuthenticationManager?) async throws -> any LanguageModel {
        switch self {
        case .system:
            guard #available(macOS 26, iOS 26, visionOS 26, *) else {
                throw ModelError.notSupported
            }
            return SystemLanguageModel()
        case .mlx(let modelId):
            return MLXLanguageModel(modelId: modelId)
        case .huggingFace(let model):
            guard let authManager = authManager else {
                throw ModelError.authenticationRequired
            }
            let token = try await authManager.getValidToken()
            return OpenAILanguageModel(
                baseURL: URL(string: "https://router.huggingface.co/v1")!,
                apiKey: token,
                model: model
            )
        }
    }
}

enum ModelError: LocalizedError {
    case authenticationRequired
    case notSupported

    var errorDescription: String? {
        switch self {
        case .authenticationRequired:
            return "Authentication required to use HuggingFace models. Please sign in."
        case .notSupported:
            return "The requested model is not supported on this OS version."
        }
    }
}
