use serde::{Deserialize, Serialize};
use serde_json::{to_writer, from_reader};
use std::fs::File;
use std::io::{Write, BufReader};
use tokio::fs::File as AsyncFile;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[derive(Debug, Serialize, Deserialize)]
pub struct Config {
    pub api_key: String,
}

pub async fn save_config(api_key: &str) -> Result<(), Box<dyn std::error::Error>> {
    let config = Config {
        api_key: api_key.to_string(),
    };

    let mut file = AsyncFile::create("config.json").await?;
    let json = serde_json::to_string(&config)?;
    file.write_all(json.as_bytes()).await?;

    Ok(())
}

pub async fn load_config() -> Result<String, Box<dyn std::error::Error>> {
    let mut file = AsyncFile::open("config.json").await?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).await?;

    let config: Config = serde_json::from_slice(&buffer)?;

    let api_key = config.api_key;

    println!("API Key: {}", api_key);

    Ok(api_key)
}
