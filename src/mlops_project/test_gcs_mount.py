import os
from pathlib import Path


def tree(path: Path, max_items: int = 200) -> None:
    if not path.exists():
        print(f"[missing] {path}")
        return

    print(f"\n=== Listing: {path.resolve()} ===")

    count = 0
    for p in sorted(path.rglob("*")):
        rel = p.relative_to(path)
        if p.is_dir():
            print(f"  [DIR ] {rel}/")
        else:
            size = p.stat().st_size
            print(f"  [FILE] {rel}  ({size} bytes)")
        count += 1
        if count >= max_items:
            print(f"  ... truncated after {max_items} items ...")
            break


def main() -> None:
    # Your bucket name from the logs
    bucket_name = os.getenv("GCS_BUCKET", "mlops-proj")

    base = Path(f"/gcs/{bucket_name}")
    data_dir = base / "data"

    print("\n=== ENV ===")
    print("PWD:", os.getcwd())
    print("USER:", os.getuid(), os.getgid())
    print("GCS_BUCKET:", bucket_name)
    print("BASE PATH:", str(base))

    # 1) list base + data folders
    tree(base)
    tree(data_dir)

    # 2) create folder inside /gcs/<bucket>/data
    out_dir = data_dir / "debug_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCreated folder: {out_dir.resolve()}")

    # 3) write a simple file
    out_file = out_dir / "hello.txt"
    out_file.write_text("hello from inside the container writing into /gcs!\n", encoding="utf-8")
    print(f"Wrote file: {out_file.resolve()}")

    # 4) list again to confirm
    tree(out_dir)
    tree(data_dir)


if __name__ == "__main__":
    main()
