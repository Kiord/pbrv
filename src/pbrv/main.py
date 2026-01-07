from importlib.resources import files as res_files

def main() -> None:
    from pbrv.app.viewer import Viewer
    from pbrv.cli import run 
    Viewer.resource_dir = str(res_files("pbrv").joinpath("resources"))
    run()

if __name__ == '__main__':
    main()