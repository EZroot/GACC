use reqwest::Client;

use crate::{config::configloader::{save_config, load_config}, clicommands::{commandparser::CommandDirectory, commands::ping}, ai::models::gpt::{chat_gpt, set_key}};

mod config{
    pub mod configloader;
}

mod clicommands {
    pub mod commandparser;

    pub mod commands {
        pub mod ping;
    }
}

mod ai {
    pub mod models {
        pub mod gpt;
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    //let key = "token";
    //save_config(key).await?; // Assuming save_config is an async function

    //Load gpt api key
    let api_key = load_config().await?;
    set_key(&api_key);

    //Register commands
    let mut command_directory = CommandDirectory::new();
    command_directory.register("ping", "p", ping::PingCommand, "This is a ping command.");
    
    let args: Vec<String> = std::env::args().collect();
    command_directory.parse_args(&args);


    //Gpt test
    let client = Client::new();

    // Call the chat_gpt function
    let system_content = "System content".to_string();
    let prompt = "User prompt".to_string();

    let (success_status, gpt_response) = chat_gpt(&client, system_content, prompt).await.unwrap();
    println!("Status: {} Response: {}", success_status, gpt_response);
    
    Ok(())
}