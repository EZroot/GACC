use serde::{Deserialize, Serialize};
use serde_json::{to_writer, from_reader};
use std::fs::File;
use std::io::{Write, BufReader};
use tokio::fs::File as AsyncFile;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub api_key: String,
    pub python_code_create_iterations:i32,
    pub python_use_conda_over_pip: bool,
}

pub async fn save_config(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    let config = Config {
        api_key: api_key.to_string(),
        python_code_create_iterations: 5,
        python_use_conda_over_pip: true,
    };

    let mut file = AsyncFile::create("config.json").await?;
    let json = serde_json::to_string(&config)?;
    file.write_all(json.as_bytes()).await?;

    Ok(())
}

pub async fn load_config() -> Result<Config, Box<dyn std::error::Error>> {
    let exe_path = std::env::current_exe()?;
    let exe_dir = exe_path.parent().expect("Failed to get executable directory");

    // Construct the path to the config file
    let config_path = exe_dir.join("config.json");
    let mut file = AsyncFile::open(config_path).await?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).await?;

    let config: Config = serde_json::from_slice(&buffer)?;

    Ok(config)
}
