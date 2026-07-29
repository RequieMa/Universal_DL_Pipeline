"""Entry point for ``python -m pipeline``.

Prints version, device information, and all registered pipeline components.
"""
from __future__ import annotations


def main() -> None:
    """Print pipeline status: version, device, registered components.

    Called when the user runs ``python -m pipeline``.
    """
    import pipeline
    from pipeline.registry import _REGISTRY
    from pipeline.utils.device import device_info, get_device

    detected = get_device("auto")
    print(f"Universal DL Pipeline v{pipeline.__version__}")
    print(f"Device: {device_info(detected)}")

    if not _REGISTRY:
        print("No components registered.")
    else:
        print("Registered components:")
        for kind in sorted(_REGISTRY):
            names = sorted(_REGISTRY[kind])
            for name in names:
                cls_name = _REGISTRY[kind][name].__name__
                print(f"  [{kind}] {name} -> {cls_name}")


if __name__ == "__main__":
    main()
