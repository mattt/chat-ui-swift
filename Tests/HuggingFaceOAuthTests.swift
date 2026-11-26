import Foundation
import Testing

@Suite struct HuggingFaceOAuthTests {
    @Test func authorizationURLRequestsOnlyNeededScopes() throws {
        let oauth = HuggingFaceOAuth(clientID: "client", redirectURI: "huggingchat://oauth/callback")
        let url = try #require(oauth.authorizationURL(codeChallenge: "challenge", state: "state"))
        let items = try #require(URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems)
        let scope = try #require(items.first(where: { $0.name == "scope" })?.value)

        #expect(scope.split(separator: " ").sorted() == ["inference-api", "openid", "profile"])
    }
}
