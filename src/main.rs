use ai::aimanager::set_key;
use codeassistant::codeassist::assist_create_code;
use reqwest::Client;

use crate::{filemanager::{configloader::{save_config, load_config}, fileloader::{create_file_python}}, clicommands::{commandparser::CommandDirectory, commands::ping}, ai::aimanager::get_prompt_by_model};

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

mod codeassistant {
    pub mod codeassist;
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

    assist_create_code("A server with the endpoint at /derp that displays a response 'hello nerd' in json", codeassistant::codeassist::CodeType::Python).await?;
    assist_create_code("A server with the endpoint at /derp that displays a response 'hello nerd' in json", codeassistant::codeassist::CodeType::Rust).await?;
    assist_create_code("A server with the endpoint at /derp that displays a response 'hello nerd' in json", codeassistant::codeassist::CodeType::CPlusPlus).await?;
    assist_create_code("A server with the endpoint at /derp that displays a response 'hello nerd' in json", codeassistant::codeassist::CodeType::CSharp).await?;

    Ok(())
}