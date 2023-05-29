use std::io::{BufWriter, stdout};

use ai::aimanager::set_key;
use clicommands::commands::pycode;
use codeassistant::codeassist::{assist_create_code, set_conda_over_pip, set_code_iterations};
use tokio::io::{self, AsyncBufReadExt, AsyncWriteExt};
use crate::{
    filemanager::{configloader::{save_config, load_config}, 
    fileloader::{create_file_python}},
     clicommands::{commandparser::CommandDirectory, commands::ping},
};

mod filemanager{
    pub mod configloader;
    pub mod fileloader;
}

mod clicommands {
    pub mod commandparser;

    pub mod commands {
        pub mod ping;
        pub mod pycode;
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
    pub mod dependencies {
        pub mod dependencychecker;
        pub mod dependants {
            pub mod python {
                pub mod mypy;
                pub mod pip;
                pub mod conda;
            }
        }
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    //let key = "token";
    //save_config(key).await?; // Assuming save_config is an async function

    //Load gpt api key
    let config_file = load_config().await?;
    let api_key = config_file.api_key;
    let conda_over_pip = config_file.python_use_conda_over_pip;
    let code_iterations = config_file.python_code_create_iterations;
    set_key(&api_key);
    set_code_iterations(code_iterations);
    set_conda_over_pip(conda_over_pip);
    //Register commands
    let mut command_directory = CommandDirectory::new();
    command_directory.register("ping", "p", ping::PingCommand, "This is a ping command.");
    command_directory.register("pycode", "pc", pycode::PythonCommand, "Creates a python file using davinci. -pc <filename> <promp>");
    
    let args: Vec<String> = std::env::args().collect();
    command_directory.parse_args(&args).await;


    // //for testing purposes
    // loop {
    //     // Read input asynchronously
    //     let stdout = io::stdout();
    //     let stdin = io::stdin();
    //     let mut writer = io::BufWriter::new(stdout);
    //     let mut reader = io::BufReader::new(stdin);
    //     let mut buffer = String::new();

    //     let message = "Python Code Promt :> ";
        
    //     // Read input asynchronously
    //     writer.write_all(message.as_bytes()).await.expect("Failed to write message");
    //     writer.flush().await.expect("Failed to flush output");
    //     reader.read_line(&mut buffer).await.expect("Failed to read line");
    
    //     assist_create_code(buffer.trim(), codeassistant::codeassist::CodeType::Python).await?;
    // }
    //assist_create_code("A server with the endpoint at /derp that displays a response 'hello nerd' in json", codeassistant::codeassist::CodeType::Rust).await?;
    //assist_create_code("A server with the endpoint at /derp that displays a response 'hello nerd' in json", codeassistant::codeassist::CodeType::CPlusPlus).await?;
    //assist_create_code("A server with the endpoint at /derp that displays a response 'hello nerd' in json", codeassistant::codeassist::CodeType::CSharp).await?;

    Ok(())
}