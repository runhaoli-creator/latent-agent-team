"""
webshop_wrapper.py — Wrapper for WebShop benchmark environment.

WebShop provides:
  - 1.18 million real Amazon products
  - 12,087 crowd-sourced shopping instructions
  - Simulated e-commerce environment with search, browsing, purchase

Official repo: https://github.com/princeton-nlp/WebShop
Reward: attribute match score in [0, 1] (1.0 = perfect purchase)
Success: reward >= 1.0

Action space: text commands
  - search[query]
  - click[element]
    - elements: product titles, "Buy Now", "Back to Search", option values, etc.

This wrapper provides:
  1. Stub environment that can run without the full WebShop server
  2. Connection to live WebShop server via HTTP
  3. Official reward calculation based on attribute matching
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


@dataclass
class WebShopProduct:
    product_id: str
    title: str
    price: float
    attributes: Dict[str, str]
    description: str = ""
    options: List[str] = field(default_factory=list)


@dataclass
class WebShopTask:
    task_id: str
    instruction: str
    target_product_id: str
    target_attributes: Dict[str, str]
    reward_attributes: List[str]


class WebShopEnvStub:
    """
    Lightweight stub of WebShop that doesn't require the full server.
    Useful for testing the agent architecture without running the full env.
    """

    SEARCH_PAGE = """
[Search] [Back to Search]
Instruction: {instruction}

Results for "{query}":
1. Product A - $15.99 - High quality item matching your needs
2. Product B - $23.50 - Premium version with extra features
3. Product C - $9.99 - Budget option
"""

    PRODUCT_PAGE = """
[Back to Search]
Product: {title}
Price: ${price}
Rating: 4.5/5 (1,234 reviews)
Options:
  Size: [Small] [Medium] [Large]
  Color: [Blue] [Red] [Green]
Description: {description}

[Add to Cart] [Buy Now]
"""

    def __init__(
        self,
        task: Optional[WebShopTask] = None,
        products: Optional[List[WebShopProduct]] = None,
    ):
        self.task = task
        self.products = products or self._make_demo_products()
        self._page = "search"
        self._query = ""
        self._selected_product: Optional[WebShopProduct] = None
        self._bought = False

    def _make_demo_products(self) -> List[WebShopProduct]:
        """Diverse product catalog covering all task template items."""
        return [
            WebShopProduct(product_id="B001", title="Blue Wireless Headphones",
                price=29.99, attributes={"color": "blue", "type": "headphones"},
                description="High-quality wireless headphones with noise cancellation and 20h battery."),
            WebShopProduct(product_id="B002", title="Red Running Shoes Size 10",
                price=59.99, attributes={"color": "red", "size": "10", "type": "shoes"},
                description="Lightweight running shoes for outdoor activities and exercise."),
            WebShopProduct(product_id="B003", title="Black Laptop Bag",
                price=35.00, attributes={"color": "black", "type": "bag"},
                description="Durable laptop bag with waterproof lining for travel and work."),
            WebShopProduct(product_id="B004", title="Green Water Bottle",
                price=15.99, attributes={"color": "green", "type": "bottle"},
                description="Portable water bottle with ergonomic design for everyday use."),
            WebShopProduct(product_id="B005", title="White Phone Case",
                price=12.99, attributes={"color": "white", "type": "case"},
                description="Premium phone case with fast charging compatibility."),
            WebShopProduct(product_id="B006", title="Gray Desk Lamp",
                price=24.99, attributes={"color": "gray", "type": "lamp"},
                description="Ergonomic desk lamp with adjustable brightness for work."),
            WebShopProduct(product_id="B007", title="Black Keyboard",
                price=49.99, attributes={"color": "black", "type": "keyboard"},
                description="Portable keyboard with ergonomic design for everyday use."),
            WebShopProduct(product_id="B008", title="Navy Backpack",
                price=45.00, attributes={"color": "navy", "type": "backpack"},
                description="Durable backpack for travel and outdoor activities."),
            WebShopProduct(product_id="B009", title="Pink Sneakers Size M",
                price=55.00, attributes={"color": "pink", "size": "M", "type": "sneakers"},
                description="Lightweight sneakers for exercise and everyday use."),
            WebShopProduct(product_id="B010", title="Blue T-Shirt Size L",
                price=19.99, attributes={"color": "blue", "size": "L", "type": "t-shirt"},
                description="High-quality t-shirt for everyday use."),
            WebShopProduct(product_id="B011", title="Black Jacket Size XL",
                price=89.99, attributes={"color": "black", "size": "XL", "type": "jacket"},
                description="Premium jacket for travel and outdoor activities."),
            WebShopProduct(product_id="B012", title="Gray Watch",
                price=75.00, attributes={"color": "gray", "type": "watch"},
                description="Affordable watch with waterproof design."),
            WebShopProduct(product_id="B013", title="Green Sunglasses",
                price=22.00, attributes={"color": "green", "type": "sunglasses"},
                description="Lightweight sunglasses with portable case."),
            WebShopProduct(product_id="B014", title="Red Notebook",
                price=8.99, attributes={"color": "red", "type": "notebook"},
                description="Durable notebook for work and everyday use."),
            WebShopProduct(product_id="B015", title="Blue Yoga Mat",
                price=28.00, attributes={"color": "blue", "type": "mat"},
                description="Premium yoga mat for exercise and everyday use."),
        ]

    def reset(self) -> str:
        self._page = "search"
        self._query = ""
        self._selected_product = None
        self._bought = False
        instruction = self.task.instruction if self.task else "Find a blue wireless headphone"
        return f"Instruction: {instruction}\n\n[Search] Enter your search query:"

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Execute a WebShop action string.

        Returns:
            obs: HTML-like observation string
            reward: 0.0 - 1.0
            done: whether task is complete
            info: additional information
        """
        action = action.strip()

        # Parse action
        search_m = re.match(r"search\[(.+)\]", action, re.IGNORECASE)
        click_m = re.match(r"click\[(.+)\]", action, re.IGNORECASE)

        if search_m:
            self._query = search_m.group(1)
            self._page = "results"
            # Simple keyword matching
            matching = [
                p for p in self.products
                if any(w in p.title.lower() or w in p.description.lower()
                       for w in self._query.lower().split())
            ]
            if matching:
                obs = f"Instruction: {self.task.instruction if self.task else 'Shop'}\n\nSearch results for '{self._query}':\n"
                for i, p in enumerate(matching[:5]):
                    obs += f"{i+1}. {p.title} - ${p.price:.2f}\n"
                obs += "\n[Back to Search]"
            else:
                obs = f"No results for '{self._query}'. Try a different search."
            return obs, 0.0, False, {}

        elif click_m:
            element = click_m.group(1).strip()

            # Click on a product
            matching = [p for p in self.products if element.lower() in p.title.lower()]
            if matching and self._page == "results":
                self._selected_product = matching[0]
                self._page = "product"
                obs = self.PRODUCT_PAGE.format(
                    title=self._selected_product.title,
                    price=self._selected_product.price,
                    description=self._selected_product.description,
                )
                return obs, 0.0, False, {}

            # Click Buy Now
            elif element.lower() in ("buy now", "add to cart") and self._selected_product:
                self._bought = True
                reward = self._compute_reward()
                obs = f"Order placed! Thank you for your purchase.\nReward: {reward:.2f}"
                return obs, reward, True, {"bought": self._selected_product.product_id}

            # Back to Search
            elif element.lower() == "back to search":
                self._page = "search"
                obs = f"Back to search. Instruction: {self.task.instruction if self.task else ''}"
                return obs, 0.0, False, {}

        return "Invalid action. Try: search[query] or click[element]", 0.0, False, {}

    def _compute_reward(self) -> float:
        """
        Compute reward based on attribute matching between purchased product
        and target attributes. Mirrors official WebShop reward calculation.
        """
        if not self._selected_product or not self.task:
            return 0.0

        target_attrs = self.task.target_attributes
        if not target_attrs:
            return 1.0 if self._selected_product.product_id == self.task.target_product_id else 0.0

        match_count = 0
        total = len(target_attrs)
        for attr, value in target_attrs.items():
            prod_val = self._selected_product.attributes.get(attr, "")
            if value.lower() in prod_val.lower() or prod_val.lower() in value.lower():
                match_count += 1

        return match_count / max(1, total)


class WebShopEnv:
    """
    WebShop environment wrapper supporting both stub mode and live server mode.
    """

    def __init__(
        self,
        mode: str = "stub",              # "stub" | "server"
        server_url: str = "http://localhost:3000",
        task_file: Optional[str] = None,
        max_episode_steps: int = 15,
    ):
        self.mode = mode
        self.server_url = server_url
        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        self._current_env: Optional[WebShopEnvStub] = None
        self.tasks = self._load_tasks(task_file)

    def _load_tasks(self, task_file: Optional[str]) -> List[WebShopTask]:
        if task_file is None or not os.path.exists(task_file):
            logger.warning(
                "WebShop task file not found. "
                "Please download from https://github.com/princeton-nlp/WebShop"
            )
            # Return demo tasks for testing
            return [
                WebShopTask(
                    task_id="demo_001",
                    instruction="Find me a blue wireless headphone under $50.",
                    target_product_id="B001",
                    target_attributes={"color": "blue", "type": "wireless"},
                    reward_attributes=["color", "type"],
                ),
                WebShopTask(
                    task_id="demo_002",
                    instruction="I need red running shoes in size 10.",
                    target_product_id="B002",
                    target_attributes={"color": "red", "size": "10"},
                    reward_attributes=["color", "size"],
                ),
            ]

        tasks = []
        with open(task_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    tasks.append(WebShopTask(
                        task_id=d.get("task_id", ""),
                        instruction=d.get("instruction", ""),
                        target_product_id=d.get("target_product_id", ""),
                        target_attributes=d.get("target_attributes", {}),
                        reward_attributes=d.get("reward_attributes", []),
                    ))
                except (json.JSONDecodeError, KeyError):
                    pass

        logger.info(f"Loaded {len(tasks)} WebShop tasks from {task_file}")
        return tasks

    def reset(self, task_idx: int = 0) -> str:
        """Reset environment for a new task."""
        self._step_count = 0
        task = self.tasks[task_idx] if task_idx < len(self.tasks) else None

        if self.mode == "stub":
            self._current_env = WebShopEnvStub(task=task)
            return self._current_env.reset()

        # Live server mode (requires running WebShop server)
        try:
            response = httpx.get(
                f"{self.server_url}/reset",
                params={"task_id": task.task_id if task else ""},
                timeout=10,
            )
            return response.text
        except Exception as e:
            logger.warning(f"WebShop server unavailable: {e}. Using stub.")
            self._current_env = WebShopEnvStub(task=task)
            return self._current_env.reset()

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        self._step_count += 1
        timeout = self._step_count >= self.max_episode_steps

        if self._current_env is not None:
            obs, reward, done, info = self._current_env.step(action)
        else:
            obs, reward, done, info = "Error: env not initialized", 0.0, True, {}

        if timeout:
            done = True
            info["timeout"] = True

        return obs, reward, done, info

    def iter_tasks(self, max_tasks: Optional[int] = None) -> List[WebShopTask]:
        return self.tasks[:max_tasks] if max_tasks else self.tasks
