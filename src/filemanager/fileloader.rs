use std::env;
use std::path::PathBuf;
use tokio::fs;
use tokio::fs::File;
use tokio::io::{self, AsyncWriteExt};

pub async fn create_file_python(filename: &str, body: &str) -> std::io::Result<String>{
    let file_name = filename;
    let file_suffex = ".py";
    let file_path = create_file(body, file_name, file_suffex).await?;
    Ok(file_path)
}

pub async fn create_file_rust(filename: &str,body: &str) -> std::io::Result<String>{
    let file_name = filename;
    let file_suffex = ".rs";
    let file_path = create_file(body, file_name, file_suffex).await?;
    Ok(file_path)
}

pub async fn create_file_csharp(filename: &str,body: &str) -> std::io::Result<String>{
    let file_name = filename;
    let file_suffex = ".cs";
    let file_path = create_file(body, file_name, file_suffex).await?;
    Ok(file_path)
}

pub async fn create_file_cplusplus(filename: &str,body: &str) -> std::io::Result<String>{
    let file_name = filename;
    let file_suffex = ".cpp";
    let file_path = create_file(body, file_name, file_suffex).await?;
    Ok(file_path)
}

async fn create_file(body: &str, file_name: &str, file_suffix: &str) -> std::io::Result<String> {
    
    let current_dir = env::current_dir().expect("Failed to get current directory");
    let curr_dir_string = current_dir.to_string_lossy().to_string();
    let file_path = format!("{}/{}{}", curr_dir_string, file_name, file_suffix);

    // Open the file asynchronously
    let mut file = File::create(&file_path).await?;

    // Write the bytes of the body to the file asynchronously
    file.write_all(body.trim().as_bytes()).await?;

    println!("Created: {}", file_path);
    Ok(file_path)
}

async fn load_file(file_path: &str) -> Result<String, std::io::Error> {
    let file_content = fs::read_to_string(file_path).await?;
    Ok(file_content)
}

pub async fn get_files_type(suffix: &str) -> io::Result<Vec<String>> {
    let mut files = Vec::new();

    let mut entries = fs::read_dir(".").await?;
    while let Some(entry) = entries.next_entry().await? {
        let path: PathBuf = entry.path();
        if path.is_file() {
            if let Some(extension) = path.extension() {
                if extension == suffix {
                    files.push(path.display().to_string());
                }
            }
        }
    }

    Ok(files)
}