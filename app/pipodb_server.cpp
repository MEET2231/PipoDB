#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <thread>
#include <mutex>
#include <algorithm>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include "vectordb/DB.h"
#include "vectordb/Filter.h"
#include "vectordb/Batch.h"
#include <fstream>

namespace {

    std::string extract_json_string(const std::string& json, const std::string& key) {
        std::string search_key = "\"" + key + "\"";
        size_t key_pos = json.find(search_key);
        if (key_pos == std::string::npos) return "";

        size_t colon_pos = json.find(':', key_pos + search_key.length());
        if (colon_pos == std::string::npos) return "";

        size_t val_start = json.find('"', colon_pos + 1);
        if (val_start == std::string::npos) return "";

        size_t val_end = json.find('"', val_start + 1);
        if (val_end == std::string::npos) return "";

        return json.substr(val_start + 1, val_end - val_start - 1);
    }

    double extract_json_number(const std::string& json, const std::string& key, double default_val = 0.0) {
        std::string search_key = "\"" + key + "\"";
        size_t key_pos = json.find(search_key);
        if (key_pos == std::string::npos) return default_val;

        size_t colon_pos = json.find(':', key_pos + search_key.length());
        if (colon_pos == std::string::npos) return default_val;

        size_t val_start = json.find_first_of("0123456789-.", colon_pos + 1);
        if (val_start == std::string::npos) return default_val;

        size_t val_end = json.find_first_not_of("0123456789-.", val_start);
        if (val_end == std::string::npos) val_end = json.length();

        try {
            return std::stod(json.substr(val_start, val_end - val_start));
        } catch (...) {
            return default_val;
        }
    }

    std::vector<float> extract_json_vector(const std::string& json, const std::string& key) {
        std::string search_key = "\"" + key + "\"";
        size_t key_pos = json.find(search_key);
        if (key_pos == std::string::npos) return {};

        size_t arr_start = json.find('[', key_pos + search_key.length());
        if (arr_start == std::string::npos) return {};

        size_t arr_end = json.find(']', arr_start + 1);
        if (arr_end == std::string::npos) return {};

        std::string arr_str = json.substr(arr_start + 1, arr_end - arr_start - 1);
        std::stringstream ss(arr_str);
        std::vector<float> result;
        std::string val;
        while (std::getline(ss, val, ',')) {
            size_t first = val.find_first_not_of(" \t\n\r");
            size_t last = val.find_last_not_of(" \t\n\r");
            if (first != std::string::npos && last != std::string::npos) {
                try {
                    result.push_back(std::stof(val.substr(first, last - first + 1)));
                } catch (...) {}
            }
        }
        return result;
    }

    std::string build_http_response(int status_code, const std::string& body, const std::string& content_type = "application/json") {
        std::stringstream ss;
        std::string status_text = (status_code == 200) ? "OK" : ((status_code == 400) ? "Bad Request" : "Internal Server Error");
        ss << "HTTP/1.1 " << status_code << " " << status_text << "\r\n";
        ss << "Content-Type: " << content_type << "\r\n";
        ss << "Content-Length: " << body.length() << "\r\n";
        ss << "Access-Control-Allow-Origin: *\r\n";
        ss << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
        ss << "Access-Control-Allow-Headers: Content-Type\r\n";
        ss << "Connection: close\r\n\r\n";
        ss << body;
        return ss.str();
    }

    void handle_client(int client_fd, vectordb::Database& db) {
        char buffer[8192];
        memset(buffer, 0, sizeof(buffer));
        ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        if (bytes_read <= 0) {
            close(client_fd);
            return;
        }

        std::string request(buffer);
        std::stringstream req_stream(request);
        std::string method, path, protocol;
        req_stream >> method >> path >> protocol;

        if (method == "OPTIONS") {
            std::string resp = build_http_response(200, "{}");
            write(client_fd, resp.c_str(), resp.length());
            close(client_fd);
            return;
        }

        // Find JSON payload body in POST requests
        std::string body = "";
        size_t body_pos = request.find("\r\n\r\n");
        if (body_pos != std::string::npos) {
            body = request.substr(body_pos + 4);
        }

        std::string response_json = "";
        int status_code = 200;

        if (path == "/health" && method == "GET") {
            response_json = "{\"status\": \"ok\", \"server\": \"PipoDB\", \"version\": \"1.0.0\"}";
        } 
        else if (path == "/api/v1/collections" && method == "GET") {
            auto cols = db.list_collections();
            std::stringstream ss;
            ss << "{\"collections\": [";
            for (size_t i = 0; i < cols.size(); ++i) {
                auto col = db.get_collection(cols[i]);
                ss << "{\"name\": \"" << cols[i] << "\""
                   << ", \"dimension\": " << (col ? col->dimension() : 128)
                   << ", \"vector_count\": " << (col ? col->size() : 0)
                   << ", \"metric\": \"" << (col ? col->params().metric : "L2") << "\""
                   << ", \"index_type\": \"" << (col ? col->params().index_type : "HNSW") << "\"}";
                if (i + 1 < cols.size()) ss << ", ";
            }
            ss << "], \"count\": " << cols.size() << "}";
            response_json = ss.str();
        } 
        else if (path == "/api/v1/collections/create" && method == "POST") {
            std::string name = extract_json_string(body, "name");
            size_t dim = static_cast<size_t>(extract_json_number(body, "dimension", 128));
            std::string metric = extract_json_string(body, "metric");
            if (metric.empty()) metric = "L2";
            std::string index_type = extract_json_string(body, "index_type");
            if (index_type.empty()) index_type = "HNSW";

            vectordb::CollectionParams params;
            params.name = name;
            params.dimension = dim;
            params.metric = metric;
            params.index_type = index_type;

            bool ok = db.create_collection(params);
            if (ok) {
                response_json = "{\"success\": true, \"message\": \"Collection '" + name + "' created successfully.\"}";
            } else {
                status_code = 400;
                response_json = "{\"success\": false, \"error\": \"Failed to create collection.\"}";
            }
        } 
        else if (path == "/api/v1/collections/drop" && method == "POST") {
            std::string name = extract_json_string(body, "name");
            bool ok = db.drop_collection(name);
            if (ok) {
                response_json = "{\"success\": true, \"message\": \"Collection '" + name + "' dropped.\"}";
            } else {
                status_code = 400;
                response_json = "{\"success\": false, \"error\": \"Collection not found.\"}";
            }
        } 
        else if (path == "/api/v1/vectors/insert" && method == "POST") {
            std::string collection_name = extract_json_string(body, "collection");
            vectordb::VectorID explicit_id = static_cast<vectordb::VectorID>(extract_json_number(body, "id", 0));
            std::vector<float> vec = extract_json_vector(body, "vector");
            std::string payload = extract_json_string(body, "payload");

            try {
                vectordb::VectorID assigned = db.insert_vector(collection_name, vec, payload, explicit_id);
                if (assigned != 0) {
                    response_json = "{\"success\": true, \"id\": " + std::to_string(assigned) + "}";
                } else {
                    status_code = 400;
                    response_json = "{\"success\": false, \"error\": \"Failed to insert vector into collection.\"}";
                }
            } catch (const std::exception& ex) {
                status_code = 400;
                response_json = "{\"success\": false, \"error\": \"" + std::string(ex.what()) + "\"}";
            }
        }
        else if (path == "/api/v1/vectors/upsert" && method == "POST") {
            std::string collection_name = extract_json_string(body, "collection");
            std::vector<float> vec = extract_json_vector(body, "vector");
            std::string payload = extract_json_string(body, "payload");
            float thresh = static_cast<float>(extract_json_number(body, "distance_threshold", 0.05));

            auto up_res = db.upsert_if_close(collection_name, vec, payload, thresh);
            if (up_res.success) {
                std::stringstream ss;
                ss << "{\"success\": true, \"id\": " << up_res.result.id 
                   << ", \"is_updated\": " << (up_res.result.is_updated ? "true" : "false")
                   << ", \"distance\": " << up_res.result.distance << "}";
                response_json = ss.str();
            } else {
                status_code = 400;
                response_json = "{\"success\": false, \"error\": \"" + up_res.error_message + "\"}";
            }
        }
        else if (path == "/api/v1/vectors/delete" && method == "POST") {
            std::string collection_name = extract_json_string(body, "collection");
            vectordb::VectorID id = static_cast<vectordb::VectorID>(extract_json_number(body, "id", 0));

            bool ok = db.remove_vector(collection_name, id);
            if (ok) {
                response_json = "{\"success\": true, \"message\": \"Vector ID " + std::to_string(id) + " deleted.\"}";
            } else {
                status_code = 400;
                response_json = "{\"success\": false, \"error\": \"Vector ID not found or deletion failed.\"}";
            }
        }
        else if (path == "/api/v1/vectors/search" && method == "POST") {
            std::string collection_name = extract_json_string(body, "collection");
            std::vector<float> vec = extract_json_vector(body, "vector");
            int top_k = static_cast<int>(extract_json_number(body, "top_k", 10));

            vectordb::DBQueryRequest req;
            req.collection_name = collection_name;
            req.query_vector = vec;
            req.top_k = top_k;
            req.include_payload = true;

            // Optional PayloadFilter
            std::string filter_key = extract_json_string(body, "filter_key");
            std::string filter_val = extract_json_string(body, "filter_value");
            if (!filter_key.empty() && !filter_val.empty()) {
                req.filter.must.push_back({filter_key, vectordb::FilterOp::EQ, filter_val});
            }

            auto search_res = db.search(req);
            if (search_res.success) {
                std::stringstream ss;
                ss << "{\"success\": true, \"query_time_ms\": " << search_res.query_time_ms << ", \"hits\": [";
                for (size_t i = 0; i < search_res.hits.size(); ++i) {
                    const auto& hit = search_res.hits[i];
                    ss << "{\"id\": " << hit.id << ", \"distance\": " << hit.distance 
                       << ", \"payload\": \"" << hit.payload_json << "\"}";
                    if (i + 1 < search_res.hits.size()) ss << ", ";
                }
                ss << "]}";
                response_json = ss.str();
            } else {
                status_code = 400;
                response_json = "{\"success\": false, \"error\": \"" + search_res.error_message + "\"}";
            }
        }
        else {
            status_code = 404;
            response_json = "{\"success\": false, \"error\": \"Endpoint not found.\"}";
        }

        std::string http_resp = build_http_response(status_code, response_json);
        write(client_fd, http_resp.c_str(), http_resp.length());
        close(client_fd);
    }

} // namespace

int main(int argc, char* argv[]) {
    int port = 8080;
    if (argc > 1) {
        port = std::stoi(argv[1]);
    }

    std::string db_dir = "./pipodb_server_data";
    vectordb::Database db(db_dir);
    db.open();

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "[ERROR] Failed to create socket." << std::endl;
        return 1;
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port);

    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "[ERROR] Failed to bind to port " << port << std::endl;
        return 1;
    }

    if (listen(server_fd, 128) < 0) {
        std::cerr << "[ERROR] Listen failed." << std::endl;
        return 1;
    }

    std::cout << "==================================================" << std::endl;
    std::cout << "     PipoDB Remote Network Server Started         " << std::endl;
    std::cout << "==================================================" << std::endl;
    std::cout << "  - Listening on : http://0.0.0.0:" << port << std::endl;
    std::cout << "  - Data Directory: " << db_dir << std::endl;
    std::cout << "  - Status       : READY" << std::endl;
    std::cout << "==================================================" << std::endl;

    while (true) {
        int client_fd = accept(server_fd, nullptr, nullptr);
        if (client_fd >= 0) {
            std::thread(handle_client, client_fd, std::ref(db)).detach();
        }
    }

    close(server_fd);
    return 0;
}
