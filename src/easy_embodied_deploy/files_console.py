"""Local model and dataset directory manager."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from .files import FileStore
from .http import read_json
from .module_http import ModuleHandler, ROOT


class Handler(ModuleHandler, BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            route = urlsplit(self.path)
            query = {k: v[-1] for k, v in parse_qs(route.query).items()}
            root = query.get("root", "models")
            if route.path == "/api/bootstrap":
                self.send(
                    {
                        "roots": [
                            {"id": n, "path": str(p)}
                            for n, p in self.server.store.roots.items()
                        ]
                    }
                )
            elif route.path == "/api/list":
                self.send(self.server.store.listing(root, query.get("path", "")))
            elif route.path == "/api/trash":
                self.send({"items": self.server.store.trash_list(root)})
            elif route.path == "/api/health":
                self.send({"ok": True})
            else:
                self.static("files")
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            self.fail(exc)

    def do_POST(self):
        try:
            body = read_json(self)
            root, path = body.get("root"), body.get("path", "")
            route = urlsplit(self.path).path
            store = self.server.store
            if route == "/api/mkdir":
                result = store.mkdir(root, path, body.get("name"))
            elif route == "/api/rename":
                result = store.rename(root, path, body.get("name"))
            elif route == "/api/delete":
                result = store.trash(root, path)
            elif route == "/api/restore":
                result = store.restore(root, body.get("id"))
            elif route == "/api/purge":
                result = store.purge(root, body.get("id"), body.get("confirmation"))
            else:
                raise ValueError("未知操作")
            self.send({"ok": True, "result": result})
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            self.fail(exc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.store = FileStore(ROOT)
    print(f"DirectoryManager: http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
