from generation_api.metrics import compute_scores


pred = ["this is a test code."]
target = ["you are so good to be true code."]

compute_scores({1: target}, {1: pred})