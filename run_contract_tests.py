from test_handler_contract import (
    test_dockerfile_pins_runtime_and_defaults,
    test_worker_is_diffusers_based_and_resident,
    test_worker_uses_presigned_urls_and_no_application_secret,
)


def main() -> None:
    test_worker_uses_presigned_urls_and_no_application_secret()
    test_worker_is_diffusers_based_and_resident()
    test_dockerfile_pins_runtime_and_defaults()
    print("Diffusers worker contract tests passed")


if __name__ == "__main__":
    main()
