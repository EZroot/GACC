use ai::aimanager::set_key;
use reqwest::Client;

use crate::{filemanager::configloader::{save_config, load_config}, clicommands::{commandparser::CommandDirectory, commands::ping}, ai::aimanager::get_prompt_by_model};

mod filemanager{
    pub mod configloader;
    pub mod fileloader;
}

mod clicommands {
    pub mod commandparser;

    pub mod commands {
        pub mod ping;
    }
}

mod ai {
    pub mod aimanager;
    
    pub mod models {
        pub mod gpt;
        pub mod davinci;
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

    let (success_status, gpt_response) = get_prompt_by_model("Write hello world in python".to_string(), ai::aimanager::AIModelType::DavinciModel).await?;
    println!("Status: {} Response: {}", success_status, gpt_response);
    println!("---DAVINCI ^ ------------------------------------------------------------------------ GPT v --------------------------");
    let (success_status, gpt_response) = get_prompt_by_model("Write hello world in python".to_string(), ai::aimanager::AIModelType::GptModel).await?;
    println!("Status: {} Response: {}", success_status, gpt_response);

    Ok(())
}