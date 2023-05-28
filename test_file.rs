

use hyper::{Body, Response, Server};
use hyper::rt::{self, Future};
use hyper::service::service_fn_ok;

fn main() {
    // Set up the server
    let addr = ([127, 0, 0, 1], 3000).into();
    let server = Server::bind(&addr)
        .serve(|| {
            // Set up the service
            service_fn_ok(|req| {
                // Handle the request
                match (req.method(), req.uri().path()) {
                    (&hyper::Method::GET, "/derp") => {
                        // Return the response
                        Response::new(Body::from(r#"{"message": "hello nerd"}"#))
                    },
                    _ => {
                        Response::builder()
                            .status(hyper::StatusCode::NOT_FOUND)
                            .body(Body::from("Not Found"))
                            .unwrap()
                    }
                }
            })
        })
        .map_err(|e| eprintln!("server error: {}", e));

    // Run the server
    println!("Listening on http://{}", addr);
    rt::run(server);
}