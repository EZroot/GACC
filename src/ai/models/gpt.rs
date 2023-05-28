use serde::{Serialize, Deserialize};
use reqwest::Response as ReqwestResponse;

#[derive(Serialize, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize, Deserialize)]
struct RequestBody {
    model: String,
    messages: Vec<Message>,
    temperature: f32,
}

#[derive(Serialize, Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Serialize, Deserialize)]
struct Choice {
    message: Message,
}

static mut API_KEY: Option<String> = None;

pub async fn chat_gpt(client: &reqwest::Client, system_content: String, prompt: String) -> Result<(bool, String), reqwest::Error> {
    let prompt = RequestBody {
        model: String::from("gpt-3.5-turbo"),
        messages: vec![
            Message { role: String::from("system"), content: system_content },
            Message { role: String::from("user"), content: "Tell me about yourself.".to_string() },
            Message { role: String::from("assistant"), content: "I'm currently kinda hungry. I like the color red. I'm near my house. I have a best friend named 'Bert'.".to_string() },
            Message { role: String::from("user"), content: prompt },
        ],
        temperature: 0.7,
    };

    let response = client.post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}",get_api_key().unwrap()))
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&prompt).unwrap())
        .send()
        .await?;

    // Check if there are any errors during the request
    if response.status().is_success() {
        let response_body = response.text().await?;
        println!("Response Body: {}", response_body);
        let res: ChatResponse = serde_json::from_str(&response_body).unwrap();

        let reply = &res.choices[0].message.content;
        let reply_as_string = reply.to_string();

        // Return success status as true and the reply
        Ok((true, reply_as_string))
    } else {
        // Return success status as false and an empty reply
        Ok((false, String::new()))
    }
}

pub fn set_key(api_key: &str) {
    unsafe {
        API_KEY = Some(api_key.to_string());
    }
}

fn get_api_key() -> Option<String> {
    unsafe { API_KEY.clone() }
}