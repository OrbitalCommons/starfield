use std::{
    collections::HashMap,
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

#[derive(Clone)]
pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}
impl Response {
    pub fn ok(body: impl Into<Vec<u8>>) -> Self {
        Self {
            status: 200,
            headers: vec![],
            body: body.into(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RecordedRequest {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
}

pub struct Stub {
    base: String,
    address: std::net::SocketAddr,
    routes: Arc<Mutex<HashMap<String, Response>>>,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
    stopping: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Stub {
    pub fn start(host: &str) -> Self {
        assert!(host == "127.0.0.1" || host == "localhost");
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let routes = Arc::new(Mutex::new(HashMap::<String, Response>::new()));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stopping = Arc::new(AtomicBool::new(false));
        let (routes2, requests2, stopping2) = (routes.clone(), requests.clone(), stopping.clone());
        let thread = thread::spawn(move || {
            for connection in listener.incoming() {
                if stopping2.load(Ordering::SeqCst) {
                    break;
                }
                let mut connection = connection.unwrap();
                connection
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut reader = BufReader::new(&mut connection);
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    continue;
                }
                let parts = line.split_whitespace().collect::<Vec<_>>();
                let method = parts[0].to_string();
                let path = parts[1].to_string();
                let mut headers = HashMap::new();
                loop {
                    line.clear();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    if let Some((name, value)) = line.split_once(':') {
                        headers.insert(name.to_ascii_lowercase(), value.trim().to_string());
                    }
                }
                let response = routes2
                    .lock()
                    .unwrap()
                    .get(&path)
                    .cloned()
                    .unwrap_or(Response {
                        status: 404,
                        headers: vec![],
                        body: vec![],
                    });
                requests2.lock().unwrap().push(RecordedRequest {
                    method: method.clone(),
                    path,
                    headers,
                });
                let _ = write!(
                    connection,
                    "HTTP/1.1 {} Stub\r\nContent-Length: {}\r\nConnection: close\r\n",
                    response.status,
                    response.body.len()
                );
                for (name, value) in response.headers {
                    let _ = write!(connection, "{name}: {value}\r\n");
                }
                let _ = write!(connection, "\r\n");
                if method != "HEAD" {
                    let _ = connection.write_all(&response.body);
                }
            }
        });
        Self {
            base: format!("http://{host}:{}", address.port()),
            address,
            routes,
            requests,
            stopping,
            thread: Some(thread),
        }
    }
    pub fn url(&self) -> String {
        self.base.clone()
    }
    pub fn route(&self, path: &str, response: Response) {
        self.routes
            .lock()
            .unwrap()
            .insert(path.to_string(), response);
    }
    pub fn requests(&self) -> Vec<RecordedRequest> {
        self.requests.lock().unwrap().clone()
    }
}
impl Drop for Stub {
    fn drop(&mut self) {
        self.stopping.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.address);
        self.thread.take().unwrap().join().unwrap();
    }
}
