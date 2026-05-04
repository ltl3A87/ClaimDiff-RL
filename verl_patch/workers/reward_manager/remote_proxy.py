import asyncio
from collections import defaultdict
import random
import threading
from typing import Any, Dict, List, Optional

import aiohttp
import requests


class SingleStepRemoteProxyManager:

    def __init__(self, rm_job, rm_num, rm_port, rm_fun):
        self.reward_server_job = rm_job
        self.reward_server_num = int(rm_num)
        self.reward_server_port = rm_port
        self.reward_server_function = rm_fun
        # Dict to track active connections per server
        self.active_connections = defaultdict(int)
        # Lock for thread-safe access to active_connections
        self.lock = threading.Lock()

        self.client_init()

    def client_init(self):

        def init_urls(job_ids, worker_num, port, path):
            urls = []
            for job_id in job_ids.split(","):
                for i in range(worker_num):
                    if i == 0:
                        index = "master-0"
                    else:
                        index = f"worker-{i - 1}"
                    url = f"http://{job_id}-{index}.{job_id}:{port}{path}"
                    urls.append(url)
            return urls

        def verify_server(url):
            """Validate if server is ready by checking the root endpoint."""
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("message") == "Reward Judge Server":
                        return True
                    else:
                        print(f"Bad response from {url}")
                        return False
                else:
                    print(f"Status {response.status_code} from {url}")
                    return False
            except Exception as e:
                print(f"Error with {url}: {e}")
                return False

        # Initialize server URLs
        all_server_ips = init_urls(self.reward_server_job, self.reward_server_num, self.reward_server_port,
                                   self.reward_server_function)

        # Verify all servers
        verified_servers = []
        for server_ip in all_server_ips:
            root_server_ip = server_ip.split(self.reward_server_function)[0]
            is_verified = verify_server(root_server_ip)
            if is_verified:
                verified_servers.append(server_ip)

        if not verified_servers:
            raise RuntimeError("No reward servers could be verified")

        self.verified_servers = verified_servers

    def maintain_load_balance(self):
        """
        Select a random server from the verified servers list.
        This method returns a randomly chosen server URL.
        """
        if not self.verified_servers:
            raise RuntimeError("No verified servers available")

        # Select a random server
        # with self.lock:
        #     selected_server = random.choice(self.verified_servers)

        #     # Still track connections for monitoring purposes
        #     self.active_connections[selected_server] += 1
        selected_server = random.choice(self.verified_servers)
        return selected_server

    def release_server(self, server_url):
        """
        Release a server connection after use.
        """
        # with self.lock:
        #     if server_url in self.active_connections:
        #         self.active_connections[server_url] = max(0, self.active_connections[server_url] - 1)
        pass

    async def _send_request_with_retry(self,
                                       session: aiohttp.ClientSession,
                                       url: str,
                                       payload: Dict[str, Any],
                                       max_retries: int = 3,
                                       timeout: int = 360) -> Optional[Dict]:
        """
        Send a request to a server with retry logic.
        
        Args:
            session: aiohttp client session
            url: Server URL
            payload: Request payload
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds
            
        Returns:
            Response data or None if all retries failed
        """
        try:
            retries = 0
            while retries < max_retries:
                try:
                    async with session.post(url, json=payload, timeout=timeout) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            print(f"Error response from {url}: {response.status}")
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    print(f"Request to {url} failed: {e}")

                retries += 1
                if retries < max_retries:
                    await asyncio.sleep(1)

            return None
        finally:
            # Always release the server when done
            self.release_server(url)

    async def _get_reward_async(self, payloads: List[Dict[str, Any]]) -> List[Optional[Dict]]:
        """
        Async implementation of get_reward.
        
        Args:
            payloads: List of payload dictionaries
            
        Returns:
            List of response data dictionaries
        """
        results = []
        async with aiohttp.ClientSession() as session:
            tasks = []
            for payload in payloads:
                server_url = self.maintain_load_balance()
                task = asyncio.create_task(self._send_request_with_retry(session, server_url, payload))
                tasks.append(task)

            results = await asyncio.gather(*tasks)

        return results

    def get_reward(self, payloads: List[Dict[str, Any]]) -> List[Optional[Dict]]:
        """
        Send payloads to reward servers for evaluation.
        
        Args:
            payloads: List of payload dictionaries
            
        Returns:
            List of response data dictionaries
        """
        try:
            # Use asyncio.run instead of directly accessing the event loop
            return asyncio.run(self._get_reward_async(payloads))
        except Exception as e:
            print(f"Error in get_reward: {e}")
            return [None] * len(payloads)


if __name__ == "__main__":
    agent_manager = SingleStepRemoteProxyManager(
        rm_job="your-reward-server-host",
        rm_num=1,
        rm_port=8000,
        rm_fun="/judge",
    )
    print(agent_manager.verified_servers)
