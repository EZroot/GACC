use crate::clicommands::commandparser::Command;

#[derive(Clone)]
pub struct PingCommand;

impl Command for PingCommand {
    fn execute(&self, args: &[String]) {
        println!("Executing 'ping' command with args: {:?}", args);
    }
}
