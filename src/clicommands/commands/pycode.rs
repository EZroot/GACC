use crate::{
    clicommands::commandparser::Command,
    codeassistant::{self, codeassist::assist_create_code},
};
use async_trait::async_trait;

#[derive(Clone)]
pub struct PythonCommand;

#[async_trait]
impl Command for PythonCommand {
    async fn execute(&self, args: &[String]) {
        if let Some((file_name, prompt)) = args.split_first() {
            let joined_string: String = prompt.join(" ");
            println!("FileName: {} Prompt: {}", file_name, joined_string);
            assist_create_code(
                file_name,
                joined_string.trim(),
                codeassistant::codeassist::CodeType::Python,
            )
            .await
            .expect("Assistant failed to create any code.");
        } else {
            println!("Args invalid. -pycode <filename> <prompt>");
        }
    }
}
