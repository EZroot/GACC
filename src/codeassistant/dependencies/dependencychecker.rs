//for all dependants, check if available, download if not

async fn check_codeassistant_dependencies(){
    if !check_mypy_installed() {
        print!("mypy is not installed. Do you want to install it? (y/n): ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let choice = input.trim().to_lowercase();

        if choice == "y" {
            match install_mypy() {
                Ok(_) => println!("mypy installed successfully."),
                Err(err) => eprintln!("Failed to install mypy: {}", err),
            }
        } else {
            println!("mypy installation skipped.");
        }
    } else {
        println!("mypy is already installed.");
    }
}