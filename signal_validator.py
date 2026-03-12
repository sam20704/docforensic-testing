# services/reasoning/signal_validator.py

def validate_signals(signals):

    validated = {
        "document_name": signals["document_name"],
        "overall_score": signals["overall_score"],
        "priority": signals["priority"],
        "modules": signals["interpreted_modules"]
    }

    return validated
