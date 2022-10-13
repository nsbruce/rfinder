from tensorboard import program  # type: ignore


def view() -> None:
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", "logs/hparam_tuning"])
    url = tb.launch()
    print(f"Tensorboard launched at {url}")
    input("Press enter to exit")
