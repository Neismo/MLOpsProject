from locust import HttpUser, task, between, events
import random
import logging


class AbstractUser(HttpUser):
    """
    Simulates a user interacting with the MLOps Abstract Embedding API.
    Tasks include searching with different query lengths, raw embedding generation,
    and health checks to simulate realistic load patterns.
    """

    wait_time = between(0.5, 2.5)

    # Sample abstracts for load testing
    abstracts = [
        "Machine learning techniques for natural language processing and text classification.",
        "Deep neural networks for computer vision and image recognition tasks.",
        "Reinforcement learning algorithms for robotics and autonomous systems.",
        "Graph neural networks for molecular property prediction in drug discovery.",
        "Transformer architectures for sequence-to-sequence modeling and translation.",
        "Federated learning for privacy-preserving distributed machine learning.",
        "Attention mechanisms in neural networks for improved model interpretability.",
        "Generative adversarial networks for image synthesis and data augmentation.",
        "Transfer learning approaches for low-resource natural language understanding.",
        "Bayesian optimization for hyperparameter tuning in deep learning models.",
        "We propose a new method for efficient neural network training using adaptive learning rates.",
        "This paper explores the impact of batch size on the convergence of stochastic gradient descent.",
        "A survey of recent advances in computer vision, focusing on object detection and segmentation.",
        "Natural language understanding with large language models: challenges and opportunities.",
        "Ethical considerations in the deployment of artificial intelligence systems in healthcare.",
        "Quantum computing algorithms for solving optimization problems in finance.",
        "Blockchain technology for secure and transparent supply chain management.",
        "Internet of Things (IoT) security: threats, vulnerabilities, and countermeasures.",
        "Edge computing architectures for real-time data processing in smart cities.",
        "5G network slicing for supporting diverse service requirements in industrial automation.",
    ]

    @task(5)
    def search_short_query(self):
        """Search with a short, simple abstract."""
        abstract = random.choice(self.abstracts[:5])
        self.client.post("/search", json={"abstract": abstract, "k": 5}, name="/search (short)")

    @task(3)
    def search_long_query(self):
        """Search with a longer, more complex abstract."""
        # Combine two abstracts to make a longer query
        a1 = random.choice(self.abstracts)
        a2 = random.choice(self.abstracts)
        abstract = f"{a1} {a2}"
        self.client.post("/search", json={"abstract": abstract, "k": 10}, name="/search (long)")

    @task(2)
    def embed_only(self):
        """Just generate embeddings without searching."""
        abstract = random.choice(self.abstracts)
        self.client.post("/embed", json={"abstract": abstract}, name="/embed")

    @task(1)
    def health_check(self):
        """Periodic health check."""
        self.client.get("/health", name="/health")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logging.info("Starting load test...")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    logging.info("Load test completed.")
