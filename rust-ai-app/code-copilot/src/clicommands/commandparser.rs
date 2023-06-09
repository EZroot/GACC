use std::{collections::HashMap, process::exit};
use async_trait::async_trait;

use std::env;
use tokio::process::{Command as TokioCommand};


// Trait representing a command
#[async_trait]
pub trait Command: CommandClone {
    async fn execute(&self, args: &[String]);
}

pub trait CommandClone {
    fn clone_box(&self) -> Box<dyn Command>;
}

impl<T: 'static + Command + Clone> CommandClone for T {
    fn clone_box(&self) -> Box<dyn Command> {
        Box::new(self.clone())
    }
}

// Command information struct
struct CommandInfo {
    command: Box<dyn Command>,
    description: String,
}

// Command directory struct
pub struct CommandDirectory {
    commands: HashMap<String, CommandInfo>,
}

impl CommandDirectory {
    pub fn new() -> Self {
        CommandDirectory {
            commands: HashMap::new(),
        }
    }

    pub fn register<C>(&mut self, command_name: &str, short_name: &str, command: C, description: &str)
    where
        C: 'static + Command + Clone,
    {
        let command_info = CommandInfo {
            command: command.clone_box(),
            description: description.to_string(),
        };

        self.commands.insert(command_name.to_string(), command_info);

        let short_command_info = CommandInfo {
            command: command.clone_box(),
            description: description.to_string(),
        };

        self.commands.insert(short_name.to_string(), short_command_info);
    }

    pub async fn parse_args(&self, args: &[String]) {
        if args.len() < 2 {
            println!("Usage: cargo run -- <command> <args...>");
            return;
        }

        let mut command: Option<&String> = None;
        let mut command_args: Vec<String> = vec![];

        for arg in args.iter().skip(1) {
            if arg.starts_with('-') {
                // We encountered a new command, execute the previous one (if any)
                if let Some(cmd) = command {
                    if let Some(command_info) = self.commands.get(&cmd[1..]) {
                        if command_args.is_empty() {
                            println!("{}", command_info.description);
                        } else {
                            command_info.command.execute(&command_args).await;
                        }
                    } else {
                        println!("Command '{}' not found", cmd);
                    }
                }

                command = Some(arg);
                command_args.clear();
            } else {
                // Add the argument to the current command's arguments
                command_args.push(arg.clone());
            }
        }

        // Execute the last command (if any)
        if let Some(cmd) = command {
            if let Some(command_info) = self.commands.get(&cmd[1..]) {
                if command_args.is_empty() {
                    println!("{}", command_info.description);
                } else {
                    command_info.command.execute(&command_args).await;
                }
            } else {
                println!("Command '{}' not found", cmd);
            }
        }
    }
}

pub async fn restart_with_args(args: &[String]) {
    let executable = env::current_exe().expect("Failed to get current executable");

    // Execute the new process with the custom arguments
    let status = TokioCommand::new(executable)
        .args(args)
        .status()
        .await;

    // Check the status of the new process
    match status {
        Ok(exit_status) => {
            if !exit_status.success() {
                eprintln!("Failed to restart: {:?}", exit_status);
            }
        }
        Err(err) => {
            eprintln!("Failed to restart: {}", err);
        }
    }

    // Terminate the current process
    exit(0);
}