import SwiftUI

struct ContentView: View {
    
    @State private var inputText: String = ""
    @State private var outputText: String = "Waiting for input..."
    @ObservedObject var modelManager = ModelManager()
    
    var body: some View {
         VStack {
             TextField("Enter text", text: $inputText)
                 .textFieldStyle(RoundedBorderTextFieldStyle())
                 .padding()

             Button("Generate") {
                 Task {
                     await modelManager.generateOutput(from: inputText)
                 }
             }
             .padding()
             
             ScrollView {
                 Text(modelManager.outputText)
                     .padding()
             }
         }
     }
}

#Preview {
    ContentView()
}
