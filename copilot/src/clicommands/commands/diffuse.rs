use crate::{
    clicommands::commandparser::Command,
    ai::aimanager,
};
use async_trait::async_trait;

#[derive(Clone)]
pub struct DiffuseCommand;

#[async_trait]
impl Command for DiffuseCommand {
    async fn execute(&self, args: &[String]) {
        if args.len() > 1 {
            let joined_string: String = args.join(" ").trim().to_string();
            aimanager::get_prompt_by_model(
                joined_string,
                aimanager::AIModelType::StableDiffuseModel,
            )
            .await
            .expect("Assistant failed to create any code.");
        } else {
            println!("Args invalid. Usage: -diffuse <prompt>");
        }
    }
}
