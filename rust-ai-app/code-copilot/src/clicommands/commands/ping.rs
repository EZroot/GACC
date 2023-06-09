use crate::clicommands::commandparser::Command;
use async_trait::async_trait;

#[derive(Clone)]
pub struct PingCommand;

#[async_trait]
impl Command for PingCommand {
    async fn execute(&self, args: &[String]) {
        println!("Executing 'ping' command with args: {:?}", args);
    }
}
