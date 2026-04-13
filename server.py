# vast_mcp_server.py
from mcp.server.fastmcp import FastMCP, Context
import requests
import json
import os
import re
import logging
from pathlib import Path
import paramiko
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager
from typing import AsyncIterator
import time
import uuid
import shlex
import posixpath
from urllib.parse import quote_plus

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VastMCPServer")

# Default configuration
DEFAULT_SERVER_URL = "https://console.vast.ai"


def _load_env_file():
    """Load .env file from project directory."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value


# Load env file before reading config
_load_env_file()

VAST_API_KEY = os.getenv("VAST_API_KEY")
SSH_KEY_FILE = (
    os.path.expanduser(os.getenv("SSH_KEY_FILE", ""))
    if os.getenv("SSH_KEY_FILE")
    else ""
)
SSH_KEY_PUBLIC_FILE = (
    os.path.expanduser(os.getenv("SSH_KEY_PUBLIC_FILE", ""))
    if os.getenv("SSH_KEY_PUBLIC_FILE")
    else ""
)
USER_NAME = os.getenv("USER_NAME", "user01")


# Validate configuration on server start, not import
def validate_configuration():
    if not VAST_API_KEY:
        raise Exception("VAST_API_KEY is not set")
    if not SSH_KEY_FILE or not os.path.exists(SSH_KEY_FILE):
        raise Exception(f"SSH_KEY_FILE does not exist: {SSH_KEY_FILE}")
    if not SSH_KEY_PUBLIC_FILE or not os.path.exists(SSH_KEY_PUBLIC_FILE):
        raise Exception(f"SSH_KEY_PUBLIC_FILE does not exist: {SSH_KEY_PUBLIC_FILE}")


# MCP Rules Configuration
class MCPRules:
    """Configuration for MCP automation rules"""

    def __init__(self):
        # Auto-attach SSH key when creating SSH/Jupyter instances
        self.auto_attach_ssh_on_create = (
            os.getenv("MCP_AUTO_ATTACH_SSH", "true").lower() == "true"
        )

        # Default instance labeling
        self.auto_label_instances = (
            os.getenv("MCP_AUTO_LABEL", "true").lower() == "true"
        )
        self.default_label_prefix = os.getenv("MCP_LABEL_PREFIX", "mcp-instance")

        # Wait for instance readiness
        self.wait_for_instance_ready = (
            os.getenv("MCP_WAIT_FOR_READY", "true").lower() == "true"
        )
        self.ready_timeout_seconds = int(
            os.getenv("MCP_READY_TIMEOUT", "300")
        )  # 5 minutes


# Global rules configuration
mcp_rules = MCPRules()


def apply_post_creation_rules(
    ctx: Context, instance_id: int, ssh: bool, jupyter: bool, original_label: str
) -> str:
    """Apply MCP rules after instance creation"""
    rule_results = []

    # Rule 1: Auto-attach SSH key for SSH/Jupyter instances
    if mcp_rules.auto_attach_ssh_on_create and (ssh or jupyter):
        try:
            ssh_result = attach_ssh(ctx, instance_id)
            rule_results.append(f"🔑 Auto SSH Key Attachment:\n{ssh_result}")
        except Exception as ssh_error:
            return f"⚠️  SSH key attachment failed: {str(ssh_error)}, try again or recreate instance"

    # Rule 2: Auto-label instance if no label provided
    if mcp_rules.auto_label_instances and not original_label:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        auto_label = f"{mcp_rules.default_label_prefix}-{timestamp}"
        try:
            label_result = label_instance(ctx, instance_id, auto_label)
            rule_results.append(f"🏷️  Auto-labeling: {label_result}")
        except Exception as label_error:
            rule_results.append(f"⚠️  Auto-labeling failed: {str(label_error)}")

    # Rule 3: Wait for instance readiness (if enabled)
    if mcp_rules.wait_for_instance_ready:
        try:
            ready_result = wait_for_instance_ready(
                ctx, instance_id, mcp_rules.ready_timeout_seconds
            )
            rule_results.append(f"⏰ Instance Readiness Check:\n{ready_result}")
        except Exception as ready_error:
            return f"⚠️ Readiness check failed: {str(ready_error)}"

    # Format results
    if rule_results:
        return (
            "\n📋 MCP Rules Applied:\n"
            + "\n".join(f"  {result}" for result in rule_results)
            + "\n"
        )
    else:
        return "\n📋 No MCP rules applied (all disabled or not applicable)\n"


def get_instance_ssh_info(ctx: Context, instance_id: int) -> tuple[str, int]:
    """Get instance ssh connection info"""
    client = get_vast_client()
    response = client._make_request(
        "GET", f"/instances/{instance_id}/", query_params={"owner": "me"}
    )
    instance = response.get("instances", {})
    ssh_host = instance.get("ssh_host")
    ssh_port = instance.get("ssh_port")
    if not ssh_host or not ssh_port:
        raise ValueError(
            f"Instance {instance_id} does not have SSH connection info available. "
            f"ssh_host={ssh_host}, ssh_port={ssh_port}"
        )
    return ssh_host, ssh_port


def wait_for_instance_ready(
    ctx: Context, instance_id: int, timeout_seconds: int = 300
) -> str:
    """Wait for instance to become ready"""
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        try:
            # Get instance details to check status
            client = get_vast_client()
            response = client._make_request(
                "GET", f"/instances/{instance_id}/", query_params={"owner": "me"}
            )

            instance = response.get("instances", {})
            status = instance.get("actual_status", "unknown")

            if status == "running":
                elapsed = time.time() - start_time
                return (
                    f"✅ Instance {instance_id} is ready! (took {elapsed:.1f} seconds)"
                )
            elif status in ["failed", "exited"]:
                return f"❌ Instance {instance_id} failed to start (status: {status})"

            # Wait before next check
            time.sleep(5)

        except Exception as e:
            logger.warning(f"Error checking instance readiness: {e}")
            time.sleep(5)

    return f"⏰ Timeout error. Now destroy instance {instance_id} and recreate similar instance again."


class VastAIClient:
    """Client for interacting with vast.ai API"""

    def __init__(self, api_key: str = None, server_url: str = None):
        self.api_key = api_key or self._load_api_key()
        self.server_url = server_url or DEFAULT_SERVER_URL
        self.session = requests.Session()

        if self.api_key:
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                }
            )

    def _load_api_key(self) -> Optional[str]:
        """Load API key from environment or file"""
        # First try environment variable
        api_key = os.getenv("VAST_API_KEY")
        if api_key:
            return api_key

        return None

    def _build_url(self, endpoint: str, query_params: Dict = None) -> str:
        """Build API URL with query parameters"""
        url = f"{self.server_url}/api/v0{endpoint}"

        if query_params:
            parts = []
            for key, value in query_params.items():
                if value is None:
                    continue
                encoded_value = quote_plus(
                    value if isinstance(value, str) else json.dumps(value)
                )
                parts.append(f"{key}={encoded_value}")
            if parts:
                url = f"{url}?{'&'.join(parts)}"

        return url

    def _make_request(
        self,
        method: str,
        endpoint: str,
        query_params: Dict = None,
        json_data: Dict = None,
    ) -> Dict:
        """Make HTTP request to vast.ai API"""
        if not self.api_key:
            raise Exception(
                "No API key configured. Set VAST_API_KEY environment variable or use 'vastai set api-key'"
            )

        url = self._build_url(endpoint, query_params)

        try:
            response = self.session.request(method, url, json=json_data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise Exception(f"Failed to connect to vast.ai API: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise Exception(f"Invalid JSON response from vast.ai API: {str(e)}")


# Global client instance
_vast_client = None


def get_vast_client() -> VastAIClient:
    """Get or create vast.ai client"""
    global _vast_client
    if not _vast_client:
        _vast_client = VastAIClient()
    return _vast_client


def parse_query_string(query_list: List[str]) -> Dict:
    """Parse query strings similar to vast CLI"""
    if not query_list:
        return {}

    query = {}
    for item in query_list:
        # Simple parsing - in real implementation this would be more complex
        if "=" in item:
            key, value = item.split("=", 1)
            try:
                # Try to convert to appropriate type
                if value.lower() == "true":
                    query[key] = {"eq": True}
                elif value.lower() == "false":
                    query[key] = {"eq": False}
                elif value.replace(".", "").isdigit():
                    query[key] = {"eq": float(value)}
                else:
                    query[key] = {"eq": value}
            except Exception:
                query[key] = {"eq": value}

    return query


def get_ssh_key(ssh_key_str: str) -> str:
    """Process SSH key string, validating and reading from file if necessary"""
    ssh_key = ssh_key_str.strip()

    # If it's a file path, read the key from the file
    if os.path.exists(ssh_key_str):
        try:
            with open(ssh_key_str, "r") as f:
                ssh_key = f.read().strip()
        except Exception as e:
            raise ValueError(
                f"Failed to read SSH key from file {ssh_key_str}: {str(e)}"
            )

    # Validate that it's not a private key
    if "PRIVATE KEY" in ssh_key:
        raise ValueError(
            "🐴 Woah, hold on there, partner!\n\n"
            "That's a *private* SSH key. You need to give the *public* one. "
            "It usually starts with 'ssh-rsa', is on a single line, has around 200 or so "
            '"base64" characters and ends with some-user@some-where.'
        )

    # Validate that it looks like an SSH public key
    if not ssh_key.lower().startswith("ssh"):
        raise ValueError(
            "Are you sure that's an SSH public key?\n\n"
            "Usually it starts with the stanza 'ssh-(keytype)' where the keytype can be "
            f"things such as rsa, ed25519-sk, or dsa. What you passed was:\n\n{ssh_key}\n\n"
            "And that just doesn't look right."
        )

    return ssh_key


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[Dict[str, Any]]:
    """Manage server startup and shutdown lifecycle"""
    try:
        logger.info("VastAI MCP server starting up")

        # Test connection to vast.ai
        try:
            client = get_vast_client()
            if client.api_key:
                logger.info("Successfully initialized vast.ai client")
            else:
                logger.warning(
                    "No API key found. Please set VAST_API_KEY environment variable or use 'vastai set api-key'"
                )
        except Exception as e:
            logger.warning(f"Could not initialize vast.ai client: {str(e)}")

        yield {}
    finally:
        logger.info("VastAI MCP server shut down")


# Add this helper function before the @mcp.tool() functions
def _execute_ssh_command(
    remote_host: str, remote_user: str, remote_port: int, command: str
) -> dict:
    """
    Helper function to execute SSH commands that can be called by other functions.
    Returns a dict with 'success', 'stdout', 'stderr', 'exit_status', and 'error' keys.
    """
    client = None
    try:
        client = _connect_ssh(remote_host, remote_user, remote_port)

        logger.info(f"Executing command: '{command}'")
        stdin, stdout, stderr = client.exec_command(command)

        stdout_output = stdout.read().decode("utf-8").strip()
        stderr_output = stderr.read().decode("utf-8").strip()
        exit_status = stdout.channel.recv_exit_status()

        return {
            "success": exit_status == 0,
            "stdout": stdout_output,
            "stderr": stderr_output,
            "exit_status": exit_status,
            "error": None,
        }

    except (FileNotFoundError, ValueError) as e:
        return {
            "success": False,
            "error": str(e),
            "stdout": "",
            "stderr": "",
            "exit_status": -1,
        }
    except paramiko.AuthenticationException:
        return {
            "success": False,
            "error": f"Authentication failed for {remote_user}@{remote_host}:{remote_port}",
            "stdout": "",
            "stderr": "",
            "exit_status": -1,
        }
    except paramiko.SSHException as e:
        return {
            "success": False,
            "error": f"SSH error occurred: {str(e)}",
            "stdout": "",
            "stderr": "",
            "exit_status": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error occurred: {str(e)}",
            "stdout": "",
            "stderr": "",
            "exit_status": -1,
        }
    finally:
        if client:
            client.close()
        logger.info("SSH connection closed")


# def _prepare_instance(host: str, port: int, user_name: str) -> str:
#     """
#     Prepare instance, create user, disable sudo password and install packages
#     Args:
#         host: str
#         port: int
#         user_name: str - user to create
#     """
#     commands = [
#         "apt update && apt upgrade -y",
#         f"useradd -m --shell /bin/bash {user_name}",
#         f"usermod -aG sudo {user_name}",
#         f"mkdir -p /home/{user_name}/.ssh",
#         f"mkdir -p /home/{user_name}/.bash_profile",
#         f"cp ~/.ssh/authorized_keys /home/{user_name}/.ssh/authorized_keys",
#         f"chown -R {user_name}:{user_name} /home/{user_name}/.ssh",
#         f"bash -c 'echo \"%sudo ALL=(ALL) NOPASSWD: ALL\" > /etc/sudoers.d/90-nopasswd-sudo'",
#         f"chmod 0440 /etc/sudoers.d/90-nopasswd-sudo"
#     ]
#
#     results = []
#     for cmd in commands:
#         result = _execute_ssh_command(host, "root", port, cmd)
#         if not result['success']:
#             raise Exception(f"❌ Failed to prepare instance at step: {cmd}\nError: {result['error']}\nSTDOUT: {result['stdout']}\nSTDERR: {result['stderr']}")
#         results.append(f"✅ {cmd}: {result['stdout']}")
#
#     results.append(f"🔒 Now you can connect: ssh -i {SSH_KEY_FILE} -p {port} {user_name}@{host}")
#
#     return f"🎉 Instance prepared successfully for user '{user_name}'!\n\n" + "\n".join(results)


def filter_templates_by_name(templates: list[dict], search_name: str) -> List[Dict]:
    """
    Filter templates by name (at least one word match).

    Args:
        api_response: The response from the vast.ai /template/ API
        search_name: The name to search for (supports partial word matching)

    Returns:
        List of templates that match at least one word in the name
    """
    if not templates:
        print("No templates found in API response")
        return []

    # Split search name into words and convert to lowercase for case-insensitive search
    search_words = [
        word.lower().strip() for word in search_name.split() if word.strip()
    ]

    if not search_words:
        print("No valid search words provided")
        return []

    # Filter templates by name word matching
    matching_templates = []
    for template in templates:
        template_name = template.get("name", "").lower()

        # Check if any search word is found in the template name
        name_matches = any(search_word in template_name for search_word in search_words)

        if name_matches:
            matching_templates.append(template)

    print(
        f"Found {len(matching_templates)} templates with name containing words from '{search_name}' out of {len(templates)} total templates"
    )

    return matching_templates


def _sftp_makedirs(sftp, remote_path):
    """Create directories recursively via SFTP"""
    dirs = []
    path = remote_path
    while len(path) > 1:
        dirs.append(path)
        path = posixpath.dirname(path)

    dirs.reverse()

    for directory in dirs:
        try:
            sftp.stat(directory)
        except FileNotFoundError:
            try:
                sftp.mkdir(directory)
            except Exception:
                pass


def _load_private_key(key_file: str):
    """Load an SSH private key from file, trying all supported key types.
    Returns the loaded key object or raises an exception.
    """
    key_types = [
        paramiko.RSAKey,
        paramiko.Ed25519Key,
        paramiko.ECDSAKey,
    ]
    try:
        key_types.append(paramiko.DSSKey)
    except AttributeError:
        pass

    last_error = None
    for key_cls in key_types:
        try:
            return key_cls.from_private_key_file(key_file)
        except paramiko.ssh_exception.PasswordRequiredException:
            raise ValueError(
                f"Private key at {key_file} is encrypted with a passphrase"
            )
        except Exception as e:
            last_error = e
            continue

    raise ValueError(f"Could not load private key from {key_file}: {last_error}")


def _connect_ssh(
    remote_host: str, remote_user: str, remote_port: int
) -> paramiko.SSHClient:
    """Create and return a connected SSH client using the configured key.
    Raises on connection failure.
    """
    if not SSH_KEY_FILE or not os.path.exists(SSH_KEY_FILE):
        raise FileNotFoundError(f"Private key file not found at: {SSH_KEY_FILE}")

    private_key = _load_private_key(SSH_KEY_FILE)

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())

    client.connect(
        hostname=remote_host,
        port=remote_port,
        username=remote_user,
        pkey=private_key,
        timeout=30,
        allow_agent=False,
        look_for_keys=False,
    )

    logger.info(
        f"SSH connection established to {remote_host}:{remote_port} as {remote_user}"
    )
    return client


# Create the MCP server
mcp = FastMCP(
    "VastAI",
    instructions="Vast.ai GPU cloud platform integration through the Model Context Protocol",
    lifespan=server_lifespan,
)


@mcp.tool()
def show_user_info(ctx: Context) -> str:
    """Show current user information and account balance"""
    try:
        client = get_vast_client()

        response = client._make_request(
            "GET", "/users/current", query_params={"owner": "me"}
        )

        user = response

        result = "User Information:\n\n"
        result += f"Username: {user.get('username', 'Unknown')}\n"
        result += f"Email: {user.get('email', 'Unknown')}\n"
        result += f"Account Balance: ${user.get('credit', 0):.2f}\n"
        result += f"User ID: {user.get('id', 'Unknown')}\n"

        if user.get("ssh_key"):
            result += "SSH Key: [configured]\n"

        # Additional account info
        if user.get("total_spent"):
            result += f"Total Spent: ${user.get('total_spent', 0):.2f}\n"

        return result

    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        return f"Error getting user info: {str(e)}"


@mcp.tool()
def show_instances(ctx: Context, owner: str = "me") -> str:
    """Show user's instances (running, stopped, etc.)"""
    if owner != "me":
        return "Error: Only 'me' is supported for the owner parameter."
    try:
        client = get_vast_client()

        response = client._make_request(
            "GET", "/instances", query_params={"owner": owner}
        )

        instances = response.get("instances", [])

        if not instances:
            return "No instances found."

        result = f"Instances ({len(instances)} total):\n\n"

        for instance in instances:
            result += f"ID: {instance.get('id', 'N/A')}\n"
            result += f"  Status: {instance.get('actual_status', 'unknown')}\n"
            result += f"  Label: {instance.get('label', 'No label')}\n"
            result += f"  Machine ID: {instance.get('machine_id', 'N/A')}\n"
            result += f"  GPU: {instance.get('gpu_name', 'N/A')}\n"
            result += f"  Cost: ${instance.get('dph_total', 0):.4f}/hour\n"
            result += f"  Image: {instance.get('image_uuid', 'N/A')}\n"
            if instance.get("public_ipaddr"):
                result += f"  IP: {instance.get('public_ipaddr')}\n"
            result += f"  Created: {instance.get('start_date', 'N/A')}\n"
            result += "\n"

        return result

    except Exception as e:
        logger.error(f"Error getting instances: {e}")
        return f"Error getting instances: {str(e)}"


@mcp.tool()
def search_offers(
    ctx: Context, query: str = "", limit: int = 20, order: str = "score-"
) -> str:
    """Search for available GPU offers/machines to rent"""
    try:
        client = get_vast_client()

        # Parse order parameter
        order_list = []
        for name in order.split(","):
            name = name.strip()
            if not name:
                continue
            direction = "asc"
            field = name
            if name.endswith("-"):
                direction = "desc"
                field = name[:-1]
            elif name.endswith("+"):
                direction = "asc"
                field = name[:-1]
            order_list.append([field, direction])

        # Build query object
        query_obj = {
            "verified": {"eq": True},
            "external": {"eq": False},
            "rentable": {"eq": True},
            "rented": {"eq": False},
            "order": order_list,
            "type": "on-demand",
            "allocated_storage": 5.0,
        }

        # Parse additional query parameters
        if query:
            query_parts = query.split()
            parsed_query = parse_query_string(query_parts)
            query_obj.update(parsed_query)

        if limit:
            query_obj["limit"] = limit

        # Use new API endpoint format
        request_data = {"select_cols": ["*"], "q": query_obj}

        response = client._make_request("PUT", "/search/asks/", json_data=request_data)

        offers = response.get("offers", [])

        if not offers:
            return "No offers found matching your criteria."

        result = f"Available Offers ({len(offers)} found):\n\n"

        for offer in offers[:limit]:
            result += f"ID: {offer.get('id', 'N/A')}\n"
            result += (
                f"  GPU: {offer.get('gpu_name', 'N/A')} x{offer.get('num_gpus', 1)}\n"
            )
            result += f"  CPU: {offer.get('cpu_name', 'N/A')}\n"
            result += f"  RAM: {offer.get('cpu_ram', 0):.1f} GB\n"
            result += f"  Disk: {offer.get('disk_space', 0):.1f} GB\n"
            result += f"  Cost: ${offer.get('dph_total', 0):.4f}/hour\n"
            result += f"  Location: {offer.get('geolocation', 'N/A')}\n"
            result += f"  Reliability: {offer.get('reliability2', 0):.1f}%\n"
            result += f"  CUDA: {offer.get('cuda_max_good', 'N/A')}\n"
            result += f"  Internet: ↓{offer.get('inet_down', 0):.0f} ↑{offer.get('inet_up', 0):.0f} Mbps\n"
            result += "\n"

        return result

    except Exception as e:
        logger.error(f"Error searching offers: {e}")
        return f"Error searching offers: {str(e)}"


@mcp.tool()
def create_instance(
    ctx: Context,
    offer_id: int,
    image: str,
    disk: float = 10.0,
    ssh: bool = False,
    jupyter: bool = False,
    direct: bool = False,
    env: dict = None,
    label: str = "",
    bid_price: float = None,
    template_id: int = None,
) -> str:
    """Create a new instance from an offer

    Parameters:
    - offer_id: ID of the offer to use for creating the instance
    - image: Docker image to run on the instance
    - disk: Amount of disk space in GB
    - ssh: Enable SSH access
    - jupyter: Enable Jupyter notebook
    - direct: Enable direct access
    - env: Environment variables dict
    - label: Label for the instance
    - bid_price: Maximum bid price per hour
    - template_id: Optional template ID to use (from search_templates)
    """
    try:
        client = get_vast_client()

        # Determine run type
        if ssh and jupyter:
            runtype = "ssh_jupyter"
        elif ssh:
            runtype = "ssh"
        elif jupyter:
            runtype = "jupyter"
        else:
            runtype = "args"

        request_data = {
            "client_id": "me",
            "image": image,
            "disk": disk,
            "ssh": ssh,
            "jupyter": jupyter,
            "direct": direct,
            "runtype": runtype,
            "label": label,
            "extra_env": env or {},
        }

        if bid_price is not None:
            request_data["price"] = bid_price

        if template_id is not None:
            request_data["template_id"] = template_id

        response = client._make_request(
            "PUT", f"/asks/{offer_id}/", json_data=request_data
        )

        if response.get("success"):
            instance_id = response.get("new_contract")
            result = f"Instance created successfully!\nInstance ID: {instance_id}\nStatus: Starting up...\n"

            # Apply MCP rules for post-creation actions
            result += apply_post_creation_rules(ctx, instance_id, ssh, jupyter, label)

            return result
        else:
            return f"Failed to create instance: {response.get('msg', 'Unknown error')}"

    except Exception as e:
        logger.error(f"Error creating instance: {e}")
        return f"Error creating instance: {str(e)}"


@mcp.tool()
def destroy_instance(ctx: Context, instance_id: int) -> str:
    """Destroy an instance, completely removing it from the system. Don't need to stop it first."""
    try:
        client = get_vast_client()

        response = client._make_request(
            "DELETE",
            f"/instances/{instance_id}/",
        )

        if response.get("success") is True:
            return f"Instance {instance_id} destroyed successfully."
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to destroy instance {instance_id}: {error_msg}"

    except Exception as e:
        logger.error(f"Error destroying instance: {e}")
        return f"Error destroying instance {instance_id}: {str(e)}"


@mcp.tool()
def start_instance(ctx: Context, instance_id: int) -> str:
    """Start a stopped instance"""
    try:
        client = get_vast_client()

        response = client._make_request(
            "PUT", f"/instances/{instance_id}/", json_data={"state": "running"}
        )

        if response.get("success") is True:
            return f"Instance {instance_id} started successfully."
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to start instance {instance_id}: {error_msg}"

    except Exception as e:
        logger.error(f"Error starting instance: {e}")
        return f"Error starting instance {instance_id}: {str(e)}"


@mcp.tool()
def stop_instance(ctx: Context, instance_id: int) -> str:
    """Stop a running instance"""
    try:
        client = get_vast_client()

        response = client._make_request(
            "PUT", f"/instances/{instance_id}/", json_data={"state": "stopped"}
        )

        if response.get("success") is True:
            return f"Instance {instance_id} stopped successfully."
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to stop instance {instance_id}: {error_msg}"

    except Exception as e:
        logger.error(f"Error stopping instance: {e}")
        return f"Error stopping instance {instance_id}: {str(e)}"


@mcp.tool()
def search_volumes(ctx: Context, query: str = "", limit: int = 20) -> str:
    """Search for available storage volume offers"""
    try:
        client = get_vast_client()

        # Default query for reliable storage
        default_query = {
            "verified": {"eq": True},
            "external": {"eq": False},
            "disk_space": {"gte": 1},
        }

        # Parse additional query parameters
        if query:
            query_parts = query.split()
            parsed_query = parse_query_string(query_parts)
            default_query.update(parsed_query)

        request_data = {
            "limit": limit,
            "allocated_storage": 1.0,
            "order": [["score", "desc"]],
        }
        request_data.update(default_query)

        response = client._make_request(
            "POST", "/volumes/search/", json_data=request_data
        )

        offers = response.get("offers", [])

        if not offers:
            return "No volume offers found matching your criteria."

        result = f"Available Volume Offers ({len(offers)} found):\n\n"

        for offer in offers[:limit]:
            result += f"ID: {offer.get('id', 'N/A')}\n"
            result += f"  Storage: {offer.get('disk_space', 0):.1f} GB\n"
            result += f"  Cost: ${offer.get('storage_cost', 0):.4f}/GB/month\n"
            result += f"  Location: {offer.get('geolocation', 'N/A')}\n"
            result += f"  Reliability: {offer.get('reliability2', 0):.1f}%\n"
            result += f"  Bandwidth: {offer.get('disk_bw', 0):.0f} MB/s\n"
            result += f"  Internet: ↓{offer.get('inet_down', 0):.0f} ↑{offer.get('inet_up', 0):.0f} Mbps\n"
            result += "\n"

        return result

    except Exception as e:
        logger.error(f"Error searching volumes: {e}")
        return f"Error searching volumes: {str(e)}"


@mcp.tool()
def label_instance(ctx: Context, instance_id: int, label: str) -> str:
    """Set a label on an instance"""
    try:
        client = get_vast_client()

        response = client._make_request(
            "PUT", f"/instances/{instance_id}/", json_data={"label": label}
        )

        if response.get("success") is True:
            return f"Label for instance {instance_id} set to '{label}'"
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to set label for instance {instance_id}: {error_msg}"

    except Exception as e:
        logger.error(f"Error setting label for instance: {e}")
        return f"Error setting label for instance {instance_id}: {str(e)}"


@mcp.tool()
def reboot_instance(ctx: Context, instance_id: int) -> str:
    """Reboot (stop/start) an instance without losing GPU priority"""
    try:
        client = get_vast_client()

        response = client._make_request(
            "PUT", f"/instances/reboot/{instance_id}/", json_data={}
        )

        if response.get("success") is True:
            return f"Instance {instance_id} is being rebooted."
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to reboot instance {instance_id}: {error_msg}"

    except Exception as e:
        logger.error(f"Error rebooting instance: {e}")
        return f"Error rebooting instance {instance_id}: {str(e)}"


@mcp.tool()
def recycle_instance(ctx: Context, instance_id: int) -> str:
    """Recycle (destroy/create) an instance from newly pulled image without losing GPU priority"""
    try:
        client = get_vast_client()

        response = client._make_request(
            "PUT", f"/instances/recycle/{instance_id}/", json_data={}
        )

        if response.get("success") is True:
            return f"Instance {instance_id} is being recycled."
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to recycle instance {instance_id}: {error_msg}"

    except Exception as e:
        logger.error(f"Error recycling instance: {e}")
        return f"Error recycling instance {instance_id}: {str(e)}"


@mcp.tool()
def show_instance(ctx: Context, instance_id: int) -> str:
    """Show detailed information about a specific instance"""
    try:
        client = get_vast_client()

        # Make request with owner param like other endpoints that work
        response = client._make_request(
            "GET", f"/instances/{instance_id}/", query_params={"owner": "me"}
        )

        # Handle error responses as per API docs
        if response.get("success") is False:
            return (
                f"Error: {response.get('msg', response.get('error', 'Unknown error'))}"
            )

        # API returns instance data in "instances" key with single object
        instance = response.get("instances", {})
        if not instance:
            return f"Instance {instance_id} not found."

        result = f"Instance {instance_id} Details:\n\n"

        # Basic status information
        result += f"Status: {instance.get('actual_status', 'unknown')}\n"
        result += f"Intended Status: {instance.get('intended_status', 'unknown')}\n"
        result += f"Current State: {instance.get('cur_state', 'unknown')}\n"
        result += f"Next State: {instance.get('next_state', 'unknown')}\n"
        result += f"Label: {instance.get('label', 'No label')}\n"

        # SSH connection info
        if instance.get("ssh_host"):
            result += f"SSH Proxy Host: {instance.get('ssh_host')}\n"
        if instance.get("ssh_port"):
            port1 = instance.get("ssh_port")
            port2 = instance.get("ssh_port") + 1
            result += f"SSH Proxy Ports: port1: {port1} or port2:{port2}\n"
        if instance.get("ssh_idx"):
            result += f"SSH Proxy Index: {instance.get('ssh_idx')}\n"

        # Network information
        if instance.get("public_ipaddr"):
            result += f"Public IP(SSH Direct IP): {instance.get('public_ipaddr')}\n"

        if instance.get("local_ipaddrs"):
            result += f"Local IPs: {', '.join(instance.get('local_ipaddrs', []))}\n"

        # Template and image info
        if instance.get("template_id"):
            result += f"Template ID: {instance.get('template_id')}\n"
        if instance.get("template_hash_id"):
            result += f"Template Hash: {instance.get('template_hash_id')}\n"
        result += f"Image UUID: {instance.get('image_uuid', 'N/A')}\n"
        if instance.get("image_args"):
            result += f"Image Args: {instance.get('image_args')}\n"
        if instance.get("image_runtype"):
            result += f"Run Type: {instance.get('image_runtype')}\n"

        # Environment and startup
        if instance.get("extra_env"):
            result += f"Extra Env: {instance.get('extra_env')}\n"
        if instance.get("onstart"):
            result += f"On Start: {instance.get('onstart')}\n"

        # Jupyter info
        if instance.get("jupyter_token"):
            token = instance.get("jupyter_token")
            masked = token[:4] + "..." if len(token) > 4 else "***"
            result += f"Jupyter Token: {masked}\n"

        # System utilization
        if instance.get("gpu_util"):
            result += f"GPU Utilization: {instance.get('gpu_util'):.1%}\n"
        if instance.get("gpu_arch"):
            result += f"GPU Architecture: {instance.get('gpu_arch')}\n"
        if instance.get("gpu_temp"):
            result += f"GPU Temperature: {instance.get('gpu_temp')}°C\n"
        if instance.get("cuda_max_good"):
            result += f"CUDA Version: {instance.get('cuda_max_good')}\n"
        if instance.get("driver_version"):
            result += f"Driver Version: {instance.get('driver_version')}\n"

        # Storage and memory
        if instance.get("disk_util"):
            result += f"Disk Utilization: {instance.get('disk_util'):.1%}\n"
        if instance.get("disk_usage"):
            result += f"Disk Usage: {instance.get('disk_usage'):.1%}\n"
        if instance.get("cpu_util"):
            result += f"CPU Utilization: {instance.get('cpu_util'):.1%}\n"
        if instance.get("mem_usage"):
            result += f"Memory Usage: {instance.get('mem_usage')} MB\n"
        if instance.get("mem_limit"):
            result += f"Memory Limit: {instance.get('mem_limit')} MB\n"
        if instance.get("vmem_usage"):
            result += f"Virtual Memory: {instance.get('vmem_usage')} MB\n"

        # Port information
        if instance.get("direct_port_start") and instance.get("direct_port_end"):
            result += f"Direct Ports: {instance.get('direct_port_start')}-{instance.get('direct_port_end')}\n"
        if instance.get("machine_dir_ssh_port"):
            result += f"Machine SSH Port: {instance.get('machine_dir_ssh_port')}\n"
        if instance.get("ports"):
            result += f"Open Ports: {instance.get('ports')}\n"

        # Runtime information
        if instance.get("uptime_mins"):
            result += f"Uptime: {instance.get('uptime_mins')} minutes\n"
        if instance.get("status_msg"):
            result += f"Status Message: {instance.get('status_msg')}\n"

        return result

    except Exception as e:
        logger.error(f"Error getting instance details: {e}")
        return f"Error getting instance {instance_id} details: {str(e)}"


@mcp.tool()
def logs(
    ctx: Context,
    instance_id: int,
    tail: str = "1000",
    filter_text: str = "",
    daemon_logs: bool = False,
) -> str:
    """Get logs for an instance"""
    try:
        client = get_vast_client()

        request_data = {}
        if filter_text:
            request_data["filter"] = filter_text
        if tail:
            request_data["tail"] = tail
        if daemon_logs:
            request_data["daemon_logs"] = "true"

        # Request logs
        response = client._make_request(
            "PUT", f"/instances/request_logs/{instance_id}/", json_data=request_data
        )

        if not response.get("result_url"):
            return f"Failed to request logs for instance {instance_id}: {response.get('msg', 'No result URL')}"

        # Poll for logs (simplified version)
        result_url = response["result_url"]

        for i in range(10):  # Reduced polling attempts for MCP
            time.sleep(0.3)
            try:
                # Make a direct request to the result URL
                log_response = client.session.get(result_url)
                if log_response.status_code == 200:
                    logs_text = log_response.text
                    if logs_text:
                        return f"Logs for instance {instance_id}:\n\n{logs_text}"
                    else:
                        return f"No logs available for instance {instance_id}"
            except Exception as log_error:
                logger.warning(f"Error fetching logs from result URL: {log_error}")
                continue

        return f"Logs for instance {instance_id} are still being prepared. Please try again in a moment."

    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return f"Error getting logs for instance {instance_id}: {str(e)}"


@mcp.tool()
def attach_ssh(ctx: Context, instance_id: int) -> str:
    """Attach an SSH key to an instance for secure access"""
    try:
        client = get_vast_client()

        with open(SSH_KEY_PUBLIC_FILE, "r") as f:
            ssh_key = f.read()

        # Process and validate the SSH key
        try:
            processed_ssh_key = get_ssh_key(ssh_key)
        except ValueError as e:
            return f"Invalid SSH key: {str(e)}"

        # Attach the SSH key to the instance
        response = client._make_request(
            "POST",
            f"/instances/{instance_id}/ssh/",
            json_data={"ssh_key": processed_ssh_key},
        )

        if response.get("success") is True:
            return f"SSH key successfully attached to instance {instance_id}. You can now connect using your private key."
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to attach SSH key to instance {instance_id}: {error_msg}"

    except Exception as e:
        logger.error(f"Error attaching SSH key: {e}")
        return f"Error attaching SSH key to instance {instance_id}: {str(e)}"


@mcp.tool()
def search_templates(ctx: Context, name_filter: str = None) -> str:
    """Search for available templates on Vast.ai"""
    """
    name_filter: str = None # Filter templates by simple name, no digits
    """
    try:
        client = get_vast_client()

        response = client._make_request(
            "GET",
            "/template/",
            query_params={
                "order_by": [{"col": "sort_order", "dir": "asc"}],
                "select_filters": {
                    "recommended": {"eq": True},
                    "private": {"eq": False},
                },
            },
        )

        if response.get("success") is False:
            return f"Failed to search templates: {response.get('msg', response.get('error', 'Unknown error'))}"

        templates = response.get("templates", [])

        # Filter templates by name
        if name_filter:
            templates = filter_templates_by_name(templates, name_filter)

        templates_found = len(templates)

        if not templates:
            return "No templates found."

        result = f"Available Templates ({templates_found} found):\n\n"

        for template in templates:
            result += f"ID: {template.get('id', 'N/A')}\n"
            result += f"  Name: {template.get('name', 'No name')}\n"
            result += f"  Image: {template.get('image', 'N/A')}\n"

            # Additional fields that might be present
            if template.get("description"):
                result += f"  Description: {template.get('description')}\n"
            if template.get("env"):
                result += f"  Environment: {template.get('env')}\n"
            if template.get("args"):
                result += f"  Arguments: {template.get('args')}\n"
            if template.get("runtype"):
                result += f"  Run Type: {template.get('runtype')}\n"
            if template.get("onstart"):
                result += f"  On Start: {template.get('onstart')}\n"
            if template.get("jupyter"):
                result += f"  Jupyter: {template.get('jupyter')}\n"
            if template.get("ssh"):
                result += f"  SSH: {template.get('ssh')}\n"

            result += "\n"

        return result

    except Exception as e:
        logger.error(f"Error searching templates: {e}")
        return f"Error searching templates: {str(e)}"


@mcp.tool()
def configure_mcp_rules(
    ctx: Context,
    auto_attach_ssh: bool = None,
    auto_label: bool = None,
    wait_for_ready: bool = None,
    label_prefix: str = None,
) -> str:
    """Configure MCP automation rules"""
    global mcp_rules

    changes = []

    if auto_attach_ssh is not None:
        mcp_rules.auto_attach_ssh_on_create = auto_attach_ssh
        changes.append(f"Auto-attach SSH: {auto_attach_ssh}")

    if auto_label is not None:
        mcp_rules.auto_label_instances = auto_label
        changes.append(f"Auto-label instances: {auto_label}")

    if wait_for_ready is not None:
        mcp_rules.wait_for_instance_ready = wait_for_ready
        changes.append(f"Wait for ready: {wait_for_ready}")

    if label_prefix is not None:
        mcp_rules.default_label_prefix = label_prefix
        changes.append(f"Label prefix: {label_prefix}")

    if changes:
        result = "⚙️  MCP Rules Configuration Updated:\n\n"
        result += "\n".join(f"  • {change}" for change in changes)
        result += "\n\nCurrent Configuration:\n"
    else:
        result = "⚙️  Current MCP Rules Configuration:\n\n"

    result += f"  • Auto-attach SSH: {mcp_rules.auto_attach_ssh_on_create}\n"
    result += f"  • Auto-label instances: {mcp_rules.auto_label_instances}\n"
    result += f"  • Label prefix: {mcp_rules.default_label_prefix}\n"
    result += f"  • Wait for ready: {mcp_rules.wait_for_instance_ready}\n"
    result += f"  • Ready timeout: {mcp_rules.ready_timeout_seconds}s\n"

    return result


# @mcp.tool()
# def execute_command(ctx: Context, instance_id: int, command: str) -> str:
#     """Execute a (constrained) remote command only available on stopped instances. Use ssh to run commands on running instances.
#
#     Available commands:
#     - ls: List directory contents
#     - rm: Remove files or directories
#     - du: Summarize device usage for a set of files
#
#     Examples:
#     - 'ls -l -o -r'
#     - 'rm -r home/delete_this.txt'
#     - 'du -d2 -h'
#     """
#     try:
#         client = get_vast_client()
#
#         # Execute the command
#         response = client._make_request(
#             "PUT",
#             f"/instances/command/{instance_id}/",
#             json_data={"command": command}
#         )
#
#         if response.get("success"):
#             result_url = response.get("result_url")
#             if not result_url:
#                 return f"Command executed but no result URL provided: {response}"
#
#             # Poll for results (simplified version for MCP)
#             for i in range(30):  # Poll up to 30 times
#                 time.sleep(0.3)
#                 try:
#                     result_response = client.session.get(result_url)
#                     if result_response.status_code == 200:
#                         output = result_response.text
#
#                         # Filter out writeable_path if provided
#                         writeable_path = response.get("writeable_path", "")
#                         if writeable_path:
#                             output = output.replace(writeable_path, "")
#
#                         return f"Command executed successfully on instance {instance_id}:\n\n{output}"
#                 except Exception as e:
#                     logger.warning(f"Error polling result URL: {e}")
#                     continue
#
#             return f"Command executed on instance {instance_id} but results are still being prepared. Please try again in a moment."
#         else:
#             return f"Failed to execute command on instance {instance_id}: {response}"
#
#     except Exception as e:
#         logger.error(f"Error executing command: {e}")
#         return f"Error executing command on instance {instance_id}: {str(e)}"


@mcp.tool()
def ssh_execute_command(
    ctx: Context, remote_host: str, remote_user: str, remote_port: int, command: str
) -> str:
    """Execute a command on a remote host via SSH

    Parameters:
    - remote_host: The hostname or IP address of the remote server
    - remote_user: The username to connect as (e.g., 'root', 'ubuntu', 'ec2-user')
    - remote_port: The SSH port number (typically 22 or custom port like 34608)
    - command: The command to execute on the remote host
    - private_key_file: Path to the SSH private key file (optional, defaults to ~/.ssh/id_rsa)

    In case connection error like "Error reading SSH protocol banner" - use port2 or direct instance ip,port instead
    """

    # Use the helper function
    result_data = _execute_ssh_command(remote_host, remote_user, remote_port, command)

    # Format result for display
    result = f"SSH Command Execution on {remote_host}:{remote_port}\n"
    result += f"Command: {command}\n"
    result += f"Exit Status: {result_data['exit_status']}\n\n"

    if result_data["stdout"]:
        result += "--- STDOUT ---\n"
        result += result_data["stdout"] + "\n\n"

    if result_data["stderr"]:
        result += "--- STDERR ---\n"
        result += result_data["stderr"] + "\n\n"

    if result_data["success"]:
        result += "✅ Command executed successfully"
    else:
        if result_data["error"]:
            result += f"❌ Command failed: {result_data['error']}"
        else:
            result += "❌ Command failed"

    return result


@mcp.tool()
def ssh_execute_background_command(
    ctx: Context,
    remote_host: str,
    remote_user: str,
    remote_port: int,
    command: str,
    task_name: str = None,
) -> str:
    """Execute a long-running command in the background on a remote host via SSH using nohup

    Parameters:
    - remote_host: The hostname or IP address of the remote server
    - remote_user: The username to connect as (e.g., 'root', 'ubuntu', 'ec2-user')
    - remote_port: The SSH port number (typically 22 or custom port like 34608)
    - command: The command to execute in the background
    - private_key_file: Path to the SSH private key file (optional, defaults to ~/.ssh/id_rsa)
    - task_name: Optional name for the task (for easier identification)

    Returns task information including task ID, process ID, and log file path

    In case connection error like "Error reading SSH protocol banner" - use port2 or direct instance ip,port instead
    """

    # Generate unique task ID
    safe_task_name = (
        re.sub(r"[^a-zA-Z0-9_\-]", "", str(task_name)) if task_name else None
    )
    task_id = str(uuid.uuid4())[:8]
    if safe_task_name:
        task_id = f"{safe_task_name}_{task_id}"

    log_file = f"/tmp/ssh_task_{task_id}.log"
    pid_file = f"/tmp/ssh_task_{task_id}.pid"

    client = None
    try:
        client = _connect_ssh(remote_host, remote_user, remote_port)

        safe_log = shlex.quote(log_file)
        safe_pid = shlex.quote(pid_file)

        bg_command = (
            f"nohup bash -c 'echo \\$\\$ > {safe_pid}; {command}' > {safe_log} 2>&1 & "
            f"sleep 0.1; if [ -f {safe_pid} ]; then cat {safe_pid}; else echo 'Failed to start'; fi"
        )

        logger.info(f"Starting background task: {task_id}")
        stdin, stdout, stderr = client.exec_command(bg_command)

        # Get the process ID
        stdout_output = stdout.read().decode("utf-8").strip()
        stderr_output = stderr.read().decode("utf-8").strip()
        exit_status = stdout.channel.recv_exit_status()

        if stderr_output or exit_status != 0:
            return f"Error starting background task:\nSTDERR: {stderr_output}\nExit Status: {exit_status}"

        try:
            process_id = int(stdout_output)
        except ValueError:
            return f"Failed to parse process ID: {stdout_output}"

        # Build result with task information
        result = f"🚀 Background Task Started Successfully!\n\n"
        result += f"Task ID: {task_id}\n"
        result += f"Process ID: {process_id}\n"
        result += f"Log File: {log_file}\n"
        result += f"PID File: {pid_file}\n"
        result += f"Command: {command}\n"
        result += f"Host: {remote_host}:{remote_port}\n\n"
        result += f"💡 Use 'ssh_check_background_task' to monitor progress\n"
        result += f"💡 Use 'ssh_kill_background_task' to stop the task\n\n"
        result += f"📝 Connection Details (save for monitoring):\n"
        result += f"   remote_host: {remote_host}\n"
        result += f"   remote_user: {remote_user}\n"
        result += f"   remote_port: {remote_port}\n"
        result += f"   task_id: {task_id}\n"
        result += f"   process_id: {process_id}"

        return result

    except Exception as e:
        return f"Error starting background task: {str(e)}"

    finally:
        if client:
            client.close()
        logger.info("SSH connection closed")


@mcp.tool()
def ssh_check_background_task(
    ctx: Context,
    remote_host: str,
    remote_user: str,
    remote_port: int,
    task_id: str,
    process_id: int,
    tail_lines: int = 50,
) -> str:
    """Check the status of a background SSH task and get its output

    Parameters:
    - remote_host: The hostname or IP address of the remote server
    - remote_user: The username to connect as
    - remote_port: The SSH port number
    - task_id: The task ID returned by ssh_execute_background_command
    - process_id: The process ID returned by ssh_execute_background_command
    - private_key_file: Path to the SSH private key file (optional)
    - tail_lines: Number of recent log lines to show (default: 50)

    In case connection error like "Error reading SSH protocol banner" - use port2 or direct instance ip,port instead
    """

    log_file = f"/tmp/ssh_task_{task_id}.log"
    pid_file = f"/tmp/ssh_task_{task_id}.pid"

    if not isinstance(process_id, int) or process_id <= 0:
        return "Error: process_id must be a positive integer"
    if not isinstance(tail_lines, int) or tail_lines < 1:
        tail_lines = 50
    tail_lines = min(tail_lines, 10000)

    client = None
    try:
        client = _connect_ssh(remote_host, remote_user, remote_port)

        safe_log = shlex.quote(log_file)
        safe_pid = shlex.quote(pid_file)
        safe_pid_int = int(process_id)
        safe_tail = int(tail_lines)

        check_process_cmd = (
            f"kill -0 {safe_pid_int} 2>/dev/null && echo 'RUNNING' || echo 'STOPPED'"
        )
        stdin, stdout, stderr = client.exec_command(check_process_cmd)
        process_status = stdout.read().decode("utf-8").strip()

        log_cmd = f"if [ -f {safe_log} ]; then tail -n {safe_tail} {safe_log}; else echo 'Log file not found'; fi"
        stdin, stdout, stderr = client.exec_command(log_cmd)
        log_content = stdout.read().decode("utf-8").strip()

        size_cmd = f"if [ -f {safe_log} ]; then wc -l {safe_log} | cut -d' ' -f1; else echo '0'; fi"
        stdin, stdout, stderr = client.exec_command(size_cmd)
        log_lines = stdout.read().decode("utf-8").strip()

        # Build status report
        result = f"📊 Background Task Status Report\n\n"
        result += f"Task ID: {task_id}\n"
        result += f"Process ID: {process_id}\n"
        result += f"Status: {'🟢 RUNNING' if process_status == 'RUNNING' else '🔴 STOPPED/COMPLETED'}\n"
        result += f"Log Lines: {log_lines}\n"
        result += f"Host: {remote_host}:{remote_port}\n\n"

        if process_status == "RUNNING":
            result += f"🔄 Task is still running...\n\n"
        else:
            result += f"✅ Task has completed or stopped\n\n"

        result += f"📄 Recent Log Output (last {tail_lines} lines):\n"
        result += f"{'=' * 50}\n"
        result += log_content
        result += f"\n{'=' * 50}\n\n"

        if process_status == "RUNNING":
            result += f"💡 Task is still running. Check again later for updates."
        else:
            result += f"💡 Task completed. Use 'ssh_execute_command' to clean up files if needed:\n"
            result += f"   rm {log_file} {pid_file}"

        return result

    except Exception as e:
        return f"Error checking background task: {str(e)}"

    finally:
        if client:
            client.close()


@mcp.tool()
def ssh_kill_background_task(
    ctx: Context,
    remote_host: str,
    remote_user: str,
    remote_port: int,
    task_id: str,
    process_id: int,
) -> str:
    """Kill a running background SSH task

    Parameters:
    - remote_host: The hostname or IP address of the remote server
    - remote_user: The username to connect as
    - remote_port: The SSH port number
    - task_id: The task ID returned by ssh_execute_background_command
    - process_id: The process ID returned by ssh_execute_background_command
    - private_key_file: Path to the SSH private key file (optional)

    In case connection error like "Error reading SSH protocol banner" - use port2 or direct instance ip,port instead
    """

    log_file = f"/tmp/ssh_task_{task_id}.log"
    pid_file = f"/tmp/ssh_task_{task_id}.pid"

    if not isinstance(process_id, int) or process_id <= 0:
        return "Error: process_id must be a positive integer"

    client = None
    try:
        client = _connect_ssh(remote_host, remote_user, remote_port)

        safe_log = shlex.quote(log_file)
        safe_pid = shlex.quote(pid_file)
        safe_pid_int = int(process_id)

        check_cmd = f"kill -0 {safe_pid_int} 2>/dev/null && echo 'RUNNING' || echo 'NOT_RUNNING'"
        stdin, stdout, stderr = client.exec_command(check_cmd)
        status = stdout.read().decode("utf-8").strip()

        if status == "NOT_RUNNING":
            result = f"⚠️ Task {task_id} (PID: {process_id}) is not running\n\n"
            result += f"The process may have already completed or been killed.\n"
        else:
            kill_cmd = (
                f"kill {safe_pid_int} 2>/dev/null; sleep 2; "
                f"if kill -0 {safe_pid_int} 2>/dev/null; then kill -9 {safe_pid_int} 2>/dev/null; echo 'FORCE_KILLED'; else echo 'TERMINATED'; fi"
            )
            stdin, stdout, stderr = client.exec_command(kill_cmd)
            kill_result = stdout.read().decode("utf-8").strip()

            result = f"🛑 Background Task Killed\n\n"
            result += f"Task ID: {task_id}\n"
            result += f"Process ID: {process_id}\n"
            result += f"Kill Result: {kill_result}\n\n"

            if kill_result == "TERMINATED":
                result += f"✅ Process terminated gracefully\n"
            elif kill_result == "FORCE_KILLED":
                result += f"✅ Process force-killed (was unresponsive)\n"
            else:
                result += f"⚠️ Unexpected result: {kill_result}\n"

        cleanup_cmd = (
            f"rm -f {safe_log} {safe_pid} 2>/dev/null; echo 'Cleanup attempted'"
        )
        stdin, stdout, stderr = client.exec_command(cleanup_cmd)
        cleanup_result = stdout.read().decode("utf-8").strip()

        result += f"\n🧹 Cleanup: {cleanup_result}\n"
        result += f"   Removed: {log_file}\n"
        result += f"   Removed: {pid_file}\n"

        return result

    except Exception as e:
        return f"Error killing background task: {str(e)}"

    finally:
        if client:
            client.close()


@mcp.tool()
def scp_upload(
    ctx: Context,
    remote_host: str,
    remote_user: str,
    remote_port: int,
    local_file_path: str,
    remote_file_path: str,
) -> str:
    """Upload a file to a remote host via SFTP (secure file transfer)

    Parameters:
    - remote_host: The hostname or IP address of the remote server
    - remote_user: The username to connect as (e.g., 'root', 'ubuntu', 'ec2-user')
    - remote_port: The SSH port number (typically 22 or custom port like 34608)
    - local_file_path: Local path to the file to upload
    - remote_file_path: Remote path where the file should be saved

    Use full path to the file, not just the filename.

    In case connection error like "Error reading SSH protocol banner" - use port2 or direct instance ip,port instead
    """

    client = None
    sftp = None

    try:
        if not os.path.exists(local_file_path):
            return f"❌ Error: Local file not found: {local_file_path}"

        local_size = os.path.getsize(local_file_path)

        client = _connect_ssh(remote_host, remote_user, remote_port)

        logger.info("SSH connection successful, starting SFTP")

        sftp = client.open_sftp()

        remote_dir = posixpath.dirname(remote_file_path)
        if remote_dir:
            try:
                _sftp_makedirs(sftp, remote_dir)
            except Exception:
                pass

        logger.info(f"Uploading {local_file_path} to {remote_file_path}")

        sftp.put(local_file_path, remote_file_path)

        # Verify upload by checking remote file size
        try:
            remote_stat = sftp.stat(remote_file_path)
            remote_size = remote_stat.st_size
        except Exception:
            remote_size = "unknown"

        result = f"📤 File Upload Successful!\n\n"
        result += f"Local File: {local_file_path}\n"
        result += f"Remote File: {remote_file_path}\n"
        result += f"Local Size: {local_size:,} bytes\n"
        result += (
            f"Remote Size: {remote_size:,} bytes\n"
            if isinstance(remote_size, int)
            else f"Remote Size: {remote_size}\n"
        )
        result += f"Host: {remote_host}:{remote_port}\n"
        result += f"User: {remote_user}\n"

        if isinstance(remote_size, int) and local_size == remote_size:
            result += "\n✅ File transfer verified successfully!"
        elif isinstance(remote_size, int):
            result += f"\n⚠️ Size mismatch detected (local: {local_size}, remote: {remote_size})"

        return result

    except FileNotFoundError:
        return f"❌ Error: Local file not found: {local_file_path}"
    except paramiko.AuthenticationException:
        return f"❌ Error: Authentication failed for {remote_user}@{remote_host}:{remote_port}"
    except paramiko.SSHException as e:
        return f"❌ Error: SSH error occurred: {str(e)}"
    except Exception as e:
        return f"❌ Error: Upload failed: {str(e)}"

    finally:
        if sftp:
            sftp.close()
        if client:
            client.close()
        logger.info("SFTP and SSH connections closed")


@mcp.tool()
def scp_download(
    ctx: Context,
    remote_host: str,
    remote_user: str,
    remote_port: int,
    remote_file_path: str,
    local_file_path: str,
) -> str:
    """Download a file from a remote host via SFTP (secure file transfer)

    Parameters:
    - remote_host: The hostname or IP address of the remote server
    - remote_user: The username to connect as (e.g., 'root', 'ubuntu', 'ec2-user')
    - remote_port: The SSH port number (typically 22 or custom port like 34608)
    - remote_file_path: Remote path to the file to download
    - local_file_path: Local path where the file should be saved

    Use full path to the file, not just the filename.

    In case connection error like "Error reading SSH protocol banner" - use port2 or direct instance ip,port instead
    """

    client = None
    sftp = None

    try:
        local_dir = os.path.dirname(local_file_path)
        if local_dir and not os.path.exists(local_dir):
            os.makedirs(local_dir, exist_ok=True)

        client = _connect_ssh(remote_host, remote_user, remote_port)

        logger.info("SSH connection successful, starting SFTP")

        sftp = client.open_sftp()

        # Check if remote file exists and get its size
        try:
            remote_stat = sftp.stat(remote_file_path)
            remote_size = remote_stat.st_size
        except FileNotFoundError:
            return f"❌ Error: Remote file not found: {remote_file_path}"
        except Exception as e:
            return f"❌ Error checking remote file: {str(e)}"

        logger.info(f"Downloading {remote_file_path} to {local_file_path}")

        # Download the file
        sftp.get(remote_file_path, local_file_path)

        # Verify download by checking local file size
        try:
            local_size = os.path.getsize(local_file_path)
        except Exception:
            local_size = "unknown"

        result = f"📥 File Download Successful!\n\n"
        result += f"Remote File: {remote_file_path}\n"
        result += f"Local File: {local_file_path}\n"
        result += f"Remote Size: {remote_size:,} bytes\n"
        result += (
            f"Local Size: {local_size:,} bytes\n"
            if isinstance(local_size, int)
            else f"Local Size: {local_size}\n"
        )
        result += f"Host: {remote_host}:{remote_port}\n"
        result += f"User: {remote_user}\n"

        if isinstance(local_size, int) and remote_size == local_size:
            result += "\n✅ File transfer verified successfully!"
        elif isinstance(local_size, int):
            result += f"\n⚠️ Size mismatch detected (remote: {remote_size}, local: {local_size})"

        return result

    except paramiko.AuthenticationException:
        return f"❌ Error: Authentication failed for {remote_user}@{remote_host}:{remote_port}"
    except paramiko.SSHException as e:
        return f"❌ Error: SSH error occurred: {str(e)}"
    except Exception as e:
        return f"❌ Error: Download failed: {str(e)}"

    finally:
        if sftp:
            sftp.close()
        if client:
            client.close()
        logger.info("SFTP and SSH connections closed")


# @mcp.tool()
# def prepare_instance(ctx: Context, instance_id: int) -> str:
#     """
#     Prepare instance, create user, disable sudo password and install packages
#     """
#     try:
#         host, port = get_instance_ssh_info(ctx, instance_id)
#         return _prepare_instance(host, port, USER_NAME)
#     except Exception as e:
#         return f"❌ Failed to prepare instance: {str(e)}"


def main():
    """Run the MCP server"""
    try:
        validate_configuration()

        import argparse

        logger.info("Starting Vast.ai MCP server")

        mcp.run()
    except Exception as e:
        logger.error(f"Failed to start MCP server: {str(e)}")
        raise


if __name__ == "__main__":
    main()
