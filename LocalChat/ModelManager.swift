import CoreML
import Foundation
import SwiftUI
import Tokenizers
import Hub

@MainActor
class ModelManager: ObservableObject {
    @Published var outputText: String = "Waiting..."
    private var model: opt125m?
    private var tokenizer: PreTrainedTokenizer?
    
    let maxOutputTokens = 30
    let endOfSequenceToken = "</s>"
    let topK: Int = 10
    
    init() {
        loadResources()
    }
    
    private func loadResources() {
        do {
            self.model = try opt125m()
            self.tokenizer = try loadTokenizer()
            print("LLM and Tokenizer loaded successfully.")
        } catch {
            DispatchQueue.main.async {
                self.outputText = "Failed to load model or tokenizer: \(error)"
            }
        }
    }
    
    private func loadTokenizer() throws -> PreTrainedTokenizer {
        guard let tokenizerConfigURL = Bundle.main.url(forResource: "tokenizer_config", withExtension: "json"),
              let tokenizerDataURL = Bundle.main.url(forResource: "tokenizer", withExtension: "json") else {
            throw NSError(domain: "App", code: 1001, userInfo: [NSLocalizedDescriptionKey: "Tokenizer files not found"])
        }
        
        let configData = try Data(contentsOf: tokenizerConfigURL)
        let tokenizerData = try Data(contentsOf: tokenizerDataURL)
        
        let config = try JSONSerialization.jsonObject(with: configData, options: []) as? [NSString: Any] ?? [:]
        let tokenizerConfig = try JSONSerialization.jsonObject(with: tokenizerData, options: []) as? [NSString: Any] ?? [:]
        
        return try AutoTokenizer.from(tokenizerConfig: Config(config), tokenizerData: Config(tokenizerConfig)) as! PreTrainedTokenizer
    }
    
    func generateOutput(from input: String) async {
        outputText = "Generating..."
        guard let model = self.model, let tokenizer = self.tokenizer else {
            outputText = "Model or Tokenizer not loaded."
            return
        }
        
        var tokenIds = tokenizer.encode(text: input, addSpecialTokens: true).map { Int32($0) }
        var generatedText = input
        
        do {
            for _ in 0..<maxOutputTokens {
                let nextTokenId = try await predictNextToken(model: model, inputTokenIds: tokenIds)
                tokenIds.append(nextTokenId)
                generatedText = tokenizer.decode(tokens: tokenIds.map { Int($0) }, skipSpecialTokens: true)
                
                if generatedText.contains(endOfSequenceToken) {
                    generatedText = generatedText.components(separatedBy: endOfSequenceToken).first ?? generatedText
                    break
                }
            }
            DispatchQueue.main.async {
                self.outputText = generatedText
            }
        } catch {
            DispatchQueue.main.async {
                self.outputText = "Inference failed: \(error.localizedDescription)"
            }
        }
    }
    
    private func predictNextToken(model: opt125m, inputTokenIds: [Int32]) async throws -> Int32 {
        let inputTensor = try MLMultiArray(shape: [1, inputTokenIds.count] as [NSNumber], dataType: .int32)
        for (i, token) in inputTokenIds.enumerated() {
            inputTensor[i] = NSNumber(value: token)
        }
        let modelInput = opt125mInput(input_ids: inputTensor)
        let modelOutput = try await model.prediction(input: modelInput)
        
        guard let logitsData = try? modelOutput.linear_72.dataPointer.assumingMemoryBound(to: Float.self) else {
            throw NSError(domain: "App", code: 1002, userInfo: [NSLocalizedDescriptionKey: "Could not access logits"])
        }
        let logitsArray = Array(UnsafeBufferPointer(start: logitsData, count: modelOutput.linear_72.count))
        return Int32(sampleFromTopK(from: logitsArray, k: topK))
    }
    
    private func sampleFromTopK(from logits: [Float], k: Int) -> Int {
        guard !logits.isEmpty else { return 0 }
        
        let indexedLogits = logits.enumerated().sorted { $0.element > $1.element }.prefix(k)
        let topKLogits = indexedLogits.map { $0.element }
        
        let exponentiatedProbs = topKLogits.map { exp($0) }
        let sumOfProbs = exponentiatedProbs.reduce(0, +)
        guard sumOfProbs > 0 else { return indexedLogits.first?.offset ?? 0 }
        
        let probabilities = exponentiatedProbs.map { $0 / sumOfProbs }
        
        var cumulativeProbability = 0.0
        let randomValue = Float.random(in: 0...1)
        
        for (i, prob) in probabilities.enumerated() {
            cumulativeProbability += Double(prob)
            if Double(randomValue) < cumulativeProbability {
                return indexedLogits[i].offset
            }
        }
        return indexedLogits.first?.offset ?? 0
    }
}
