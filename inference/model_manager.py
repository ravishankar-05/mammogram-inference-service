from .mmbcd_service import MMBCDService
from .focalnet_service import FocalNetService


class ModelManager:

    def __init__(self):
        self.current_model = None
        self.current_model_name = None

    def load_model(self, model_name):
        """
        Load only the requested model.

        If another model is already loaded,
        it will be unloaded first.
        """

        # Unload the currently loaded model
        # before loading a new one.
        if self.current_model is not None:
            self.unload_model()

        print(f"Loading model: {model_name}")

        # Load MMBCD model
        if model_name == "MMBCD":

            self.current_model = MMBCDService()
            self.current_model.load_model()

        # Load FocalNet model
        elif model_name == "FocalNet":

            self.current_model = FocalNetService()
            self.current_model.load_model()

        # Reject unsupported model names
        else:
            raise ValueError(
                f"Unsupported model: {model_name}"
            )

        # Store the name of the currently loaded model
        self.current_model_name = model_name

        print(
            f"{model_name} loaded successfully"
        )

        return self.current_model

    def unload_model(self):
        """
        Unload the currently loaded model.
        """

        if self.current_model is not None:

            # Both MMBCDService and FocalNetService
            # provide their own cleanup method.
            self.current_model.unload_model()

            self.current_model = None

        print(
            f"Unloaded model: "
            f"{self.current_model_name}"
        )

        self.current_model_name = None