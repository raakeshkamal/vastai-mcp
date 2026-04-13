from mcp.server.fastmcp import FastMCP, Context
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

from vastai import VastAI

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("VastMCPServer")


def _load_env_file():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip()
                if key and key not in os.environ:
                    os.environ[key] = value


_load_env_file()

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


def validate_configuration():
    if not SSH_KEY_FILE or not os.path.exists(SSH_KEY_FILE):
        raise Exception(f"SSH_KEY_FILE does not exist: {SSH_KEY_FILE}")
    if not SSH_KEY_PUBLIC_FILE or not os.path.exists(SSH_KEY_PUBLIC_FILE):
        raise Exception(f"SSH_KEY_PUBLIC_FILE does not exist: {SSH_KEY_PUBLIC_FILE}")


class MCPRules:
    def __init__(self):
        self.auto_attach_ssh_on_create = (
            os.getenv("MCP_AUTO_ATTACH_SSH", "true").lower() == "true"
        )
        self.auto_label_instances = (
            os.getenv("MCP_AUTO_LABEL", "true").lower() == "true"
        )
        self.default_label_prefix = os.getenv("MCP_LABEL_PREFIX", "mcp-instance")
        self.wait_for_instance_ready = (
            os.getenv("MCP_WAIT_FOR_READY", "true").lower() == "true"
        )
        self.ready_timeout_seconds = int(os.getenv("MCP_READY_TIMEOUT", "300"))


mcp_rules = MCPRules()


def apply_post_creation_rules(
    ctx: Context, instance_id: int, ssh: bool, jupyter: bool, original_label: str
) -> str:
    rule_results = []

    if mcp_rules.auto_attach_ssh_on_create and (ssh or jupyter):
        try:
            ssh_result = attach_ssh(ctx, instance_id)
            rule_results.append(f"Auto SSH Key Attachment:\n{ssh_result}")
        except Exception as ssh_error:
            return f"SSH key attachment failed: {str(ssh_error)}, try again or recreate instance"

    if mcp_rules.auto_label_instances and not original_label:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        auto_label = f"{mcp_rules.default_label_prefix}-{timestamp}"
        try:
            label_result = label_instance(ctx, instance_id, auto_label)
            rule_results.append(f"Auto-labeling: {label_result}")
        except Exception as label_error:
            rule_results.append(f"Auto-labeling failed: {str(label_error)}")

    if mcp_rules.wait_for_instance_ready:
        try:
            ready_result = wait_for_instance_ready(
                ctx, instance_id, mcp_rules.ready_timeout_seconds
            )
            rule_results.append(f"Instance Readiness Check:\n{ready_result}")
        except Exception as ready_error:
            return f"Readiness check failed: {str(ready_error)}"

    if rule_results:
        return (
            "\nMCP Rules Applied:\n"
            + "\n".join(f"  {result}" for result in rule_results)
            + "\n"
        )
    return "\nNo MCP rules applied (all disabled or not applicable)\n"


def wait_for_instance_ready(
    ctx: Context, instance_id: int, timeout_seconds: int = 300
) -> str:
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        try:
            instance = vast.show_instance(instance_id)
            status = instance.get("actual_status", "unknown")

            if status == "running":
                elapsed = time.time() - start_time
                return f"Instance {instance_id} is ready! (took {elapsed:.1f} seconds)"
            elif status in ["failed", "exited"]:
                return f"Instance {instance_id} failed to start (status: {status})"

            time.sleep(5)

        except Exception as e:
            logger.warning(f"Error checking instance readiness: {e}")
            time.sleep(5)

    return f"Timeout error. Now destroy instance {instance_id} and recreate similar instance again."


vast: Optional[VastAI] = None


def get_ssh_key(ssh_key_str: str) -> str:
    ssh_key = ssh_key_str.strip()

    if os.path.exists(ssh_key_str):
        try:
            with open(ssh_key_str, "r") as f:
                ssh_key = f.read().strip()
        except Exception as e:
            raise ValueError(
                f"Failed to read SSH key from file {ssh_key_str}: {str(e)}"
            )

    if "PRIVATE KEY" in ssh_key:
        raise ValueError(
            "Woah, hold on there, partner!\n\n"
            "That's a *private* SSH key. You need to give the *public* one. "
            "It usually starts with 'ssh-rsa', is on a single line, has around 200 or so "
            '"base64" characters and ends with some-user@some-where.'
        )

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
    global vast
    try:
        logger.info("VastAI MCP server starting up")

        try:
            api_key = os.getenv("VAST_API_KEY")
            vast = VastAI(api_key=api_key) if api_key else VastAI()
            logger.info("Successfully initialized vast.ai client")
        except Exception as e:
            logger.warning(f"Could not initialize vast.ai client: {str(e)}")

        yield {}
    finally:
        logger.info("VastAI MCP server shut down")


def _execute_ssh_command(
    remote_host: str, remote_user: str, remote_port: int, command: str
) -> dict:
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


def filter_templates_by_name(templates: list[dict], search_name: str) -> List[Dict]:
    if not templates:
        print("No templates found in API response")
        return []

    search_words = [
        word.lower().strip() for word in search_name.split() if word.strip()
    ]

    if not search_words:
        print("No valid search words provided")
        return []

    matching_templates = []
    for template in templates:
        template_name = template.get("name", "").lower()

        name_matches = any(search_word in template_name for search_word in search_words)

        if name_matches:
            matching_templates.append(template)

    print(
        f"Found {len(matching_templates)} templates with name containing words from '{search_name}' out of {len(templates)} total templates"
    )

    return matching_templates


def _sftp_makedirs(sftp, remote_path):
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
    key_types = [
        paramiko.RSAKey,
        paramiko.Ed25519Key,
        paramiko.ECDSAKey,
    ]

    last_error = None
    for key_cls in key_types:
        try:
            return key_cls.from_private_key_file(key_file)
        except Exception as e:
            last_error = e
            continue

    raise ValueError(f"Could not load private key from {key_file}: {last_error}")


def _connect_ssh(
    remote_host: str, remote_user: str, remote_port: int
) -> paramiko.SSHClient:
    if not SSH_KEY_FILE or not os.path.exists(SSH_KEY_FILE):
        raise FileNotFoundError(f"Private key file not found at: {SSH_KEY_FILE}")

    private_key = _load_private_key(SSH_KEY_FILE)

    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.RejectPolicy())

    try:
        client.connect(
            hostname=remote_host,
            port=remote_port,
            username=remote_user,
            pkey=private_key,
            timeout=30,
            allow_agent=False,
            look_for_keys=False,
        )
    except Exception as e:
        raise ValueError(
            f"Could not connect via SSH to {remote_host}:{remote_port}: {e}"
        )

    logger.info(
        f"SSH connection established to {remote_host}:{remote_port} as {remote_user}"
    )
    return client


mcp = FastMCP(
    "VastAI",
    instructions="Vast.ai GPU cloud platform integration through the Model Context Protocol",
    lifespan=server_lifespan,
)


@mcp.tool()
def show_user_info(ctx: Context) -> str:
    """Show current user information and account balance"""
    try:
        user = vast.show_user()

        result = "User Information:\n\n"
        result += f"Username: {user.get('username', 'Unknown')}\n"
        result += f"Email: {user.get('email', 'Unknown')}\n"
        result += f"Account Balance: ${user.get('credit', 0):.2f}\n"
        result += f"User ID: {user.get('id', 'Unknown')}\n"

        if user.get("ssh_key"):
            result += "SSH Key: [configured]\n"

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
        instances = vast.show_instances()

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
def show_instance(ctx: Context, instance_id: int) -> str:
    """Show detailed information about a specific instance"""
    try:
        instance = vast.show_instance(instance_id)

        result = f"Instance {instance_id} Details:\n\n"

        result += f"Status: {instance.get('actual_status', 'unknown')}\n"
        result += f"Intended Status: {instance.get('intended_status', 'unknown')}\n"
        result += f"Current State: {instance.get('cur_state', 'unknown')}\n"
        result += f"Next State: {instance.get('next_state', 'unknown')}\n"
        result += f"Label: {instance.get('label', 'No label')}\n"

        if instance.get("ssh_host"):
            result += f"SSH Proxy Host: {instance.get('ssh_host')}\n"
        if instance.get("ssh_port"):
            port1 = instance.get("ssh_port")
            port2 = instance.get("ssh_port") + 1
            result += f"SSH Proxy Ports: port1: {port1} or port2:{port2}\n"
        if instance.get("ssh_idx"):
            result += f"SSH Proxy Index: {instance.get('ssh_idx')}\n"

        if instance.get("public_ipaddr"):
            result += f"Public IP(SSH Direct IP): {instance.get('public_ipaddr')}\n"

        if instance.get("local_ipaddrs"):
            result += f"Local IPs: {', '.join(instance.get('local_ipaddrs', []))}\n"

        if instance.get("template_id"):
            result += f"Template ID: {instance.get('template_id')}\n"
        if instance.get("template_hash_id"):
            result += f"Template Hash: {instance.get('template_hash_id')}\n"
        result += f"Image UUID: {instance.get('image_uuid', 'N/A')}\n"
        if instance.get("image_args"):
            result += f"Image Args: {instance.get('image_args')}\n"
        if instance.get("image_runtype"):
            result += f"Run Type: {instance.get('image_runtype')}\n"

        if instance.get("extra_env"):
            result += f"Extra Env: {instance.get('extra_env')}\n"
        if instance.get("onstart"):
            result += f"On Start: {instance.get('onstart')}\n"

        if instance.get("jupyter_token"):
            token = instance.get("jupyter_token")
            masked = token[:4] + "..." if len(token) > 4 else "***"
            result += f"Jupyter Token: {masked}\n"

        if instance.get("gpu_util"):
            result += f"GPU Utilization: {instance.get('gpu_util'):.1%}\n"
        if instance.get("gpu_arch"):
            result += f"GPU Architecture: {instance.get('gpu_arch')}\n"
        if instance.get("gpu_temp"):
            result += f"GPU Temperature: {instance.get('gpu_temp')}C\n"
        if instance.get("cuda_max_good"):
            result += f"CUDA Version: {instance.get('cuda_max_good')}\n"
        if instance.get("driver_version"):
            result += f"Driver Version: {instance.get('driver_version')}\n"

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

        if instance.get("direct_port_start") and instance.get("direct_port_end"):
            result += f"Direct Ports: {instance.get('direct_port_start')}-{instance.get('direct_port_end')}\n"
        if instance.get("machine_dir_ssh_port"):
            result += f"Machine SSH Port: {instance.get('machine_dir_ssh_port')}\n"
        if instance.get("ports"):
            result += f"Open Ports: {instance.get('ports')}\n"

        if instance.get("uptime_mins"):
            result += f"Uptime: {instance.get('uptime_mins')} minutes\n"
        if instance.get("status_msg"):
            result += f"Status Message: {instance.get('status_msg')}\n"

        return result

    except Exception as e:
        logger.error(f"Error getting instance details: {e}")
        return f"Error getting instance {instance_id} details: {str(e)}"


@mcp.tool()
def search_offers(
    ctx: Context, query: str = "", limit: int = 20, order: str = "score-"
) -> str:
    """Search for available GPU offers/machines to rent"""
    try:
        offers = vast.search_offers(query=query, order=order, limit=limit)

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
            result += f"  Internet: {offer.get('inet_down', 0):.0f} / {offer.get('inet_up', 0):.0f} Mbps\n"
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
        if ssh and jupyter:
            runtype = "ssh_jupyter"
        elif ssh:
            runtype = "ssh"
        elif jupyter:
            runtype = "jupyter"
        else:
            runtype = "args"

        kwargs = {
            "image": image,
            "disk": disk,
            "env": env or {},
            "label": label,
            "runtype": runtype,
        }

        if bid_price is not None:
            kwargs["price"] = bid_price

        if template_id is not None:
            kwargs["template_hash"] = template_id

        response = vast.create_instance(offer_id, **kwargs)

        if response.get("success"):
            instance_id = response.get("new_contract")
            result = f"Instance created successfully!\nInstance ID: {instance_id}\nStatus: Starting up...\n"

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
        response = vast.destroy_instance(instance_id)

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
        response = vast.start_instance(instance_id)

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
        response = vast.stop_instance(instance_id)

        if response.get("success") is True:
            return f"Instance {instance_id} stopped successfully."
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to stop instance {instance_id}: {error_msg}"

    except Exception as e:
        logger.error(f"Error stopping instance: {e}")
        return f"Error stopping instance {instance_id}: {str(e)}"


@mcp.tool()
def reboot_instance(ctx: Context, instance_id: int) -> str:
    """Reboot (stop/start) an instance without losing GPU priority"""
    try:
        response = vast.reboot_instance(instance_id)

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
        response = vast.recycle_instance(instance_id)

        if response.get("success") is True:
            return f"Instance {instance_id} is being recycled."
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to recycle instance {instance_id}: {error_msg}"

    except Exception as e:
        logger.error(f"Error recycling instance: {e}")
        return f"Error recycling instance {instance_id}: {str(e)}"


@mcp.tool()
def label_instance(ctx: Context, instance_id: int, label: str) -> str:
    """Set a label on an instance"""
    try:
        response = vast.label_instance(instance_id, label)

        if response.get("success") is True:
            return f"Label for instance {instance_id} set to '{label}'"
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to set label for instance {instance_id}: {error_msg}"

    except Exception as e:
        logger.error(f"Error setting label for instance: {e}")
        return f"Error setting label for instance {instance_id}: {str(e)}"


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
        log_text = vast.logs(
            instance_id, tail=tail, filter=filter_text, daemon_logs=daemon_logs
        )

        if isinstance(log_text, str):
            return f"Logs for instance {instance_id}:\n\n{log_text}"
        else:
            return f"Logs for instance {instance_id}:\n\n{log_text}"

    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return f"Error getting logs for instance {instance_id}: {str(e)}"


@mcp.tool()
def attach_ssh(ctx: Context, instance_id: int) -> str:
    """Attach an SSH key to an instance for secure access"""
    try:
        with open(SSH_KEY_PUBLIC_FILE, "r") as f:
            ssh_key = f.read()

        try:
            processed_ssh_key = get_ssh_key(ssh_key)
        except ValueError as e:
            return f"Invalid SSH key: {str(e)}"

        response = vast.attach_ssh(instance_id, processed_ssh_key)

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
    try:
        templates = vast.search_templates()

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
def search_volumes(ctx: Context, query: str = "", limit: int = 20) -> str:
    """Search for available storage volume offers"""
    try:
        offers = vast.search_volumes(query=query if query else None, limit=limit)

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
            result += f"  Internet: {offer.get('inet_down', 0):.0f} / {offer.get('inet_up', 0):.0f} Mbps\n"
            result += "\n"

        return result

    except Exception as e:
        logger.error(f"Error searching volumes: {e}")
        return f"Error searching volumes: {str(e)}"


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
        result = "MCP Rules Configuration Updated:\n\n"
        result += "\n".join(f"  - {change}" for change in changes)
        result += "\n\nCurrent Configuration:\n"
    else:
        result = "Current MCP Rules Configuration:\n\n"

    result += f"  - Auto-attach SSH: {mcp_rules.auto_attach_ssh_on_create}\n"
    result += f"  - Auto-label instances: {mcp_rules.auto_label_instances}\n"
    result += f"  - Label prefix: {mcp_rules.default_label_prefix}\n"
    result += f"  - Wait for ready: {mcp_rules.wait_for_instance_ready}\n"
    result += f"  - Ready timeout: {mcp_rules.ready_timeout_seconds}s\n"

    return result


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

    In case connection error like "Error reading SSH protocol banner" - use port2 or direct instance ip,port instead
    """
    result_data = _execute_ssh_command(remote_host, remote_user, remote_port, command)

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
        result += "Command executed successfully"
    else:
        if result_data["error"]:
            result += f"Command failed: {result_data['error']}"
        else:
            result += "Command failed"

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

    Returns task information including task ID, process ID, and log file path.

    In case connection error like "Error reading SSH protocol banner" - use port2 or direct instance ip,port instead
    """
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
            f"nohup bash -c 'echo $$ > {safe_pid}; {command}' > {safe_log} 2>&1 & "
            f"sleep 0.1; if [ -f {safe_pid} ]; then cat {safe_pid}; else echo 'Failed to start'; fi"
        )

        logger.info(f"Starting background task: {task_id}")
        stdin, stdout, stderr = client.exec_command(bg_command)

        stdout_output = stdout.read().decode("utf-8").strip()
        stderr_output = stderr.read().decode("utf-8").strip()
        exit_status = stdout.channel.recv_exit_status()

        if stderr_output or exit_status != 0:
            return f"Error starting background task:\nSTDERR: {stderr_output}\nExit Status: {exit_status}"

        try:
            process_id = int(stdout_output)
        except ValueError:
            return f"Failed to parse process ID: {stdout_output}"

        result = f"Background Task Started Successfully!\n\n"
        result += f"Task ID: {task_id}\n"
        result += f"Process ID: {process_id}\n"
        result += f"Log File: {log_file}\n"
        result += f"PID File: {pid_file}\n"
        result += f"Command: {command}\n"
        result += f"Host: {remote_host}:{remote_port}\n\n"
        result += f"Use 'ssh_check_background_task' to monitor progress\n"
        result += f"Use 'ssh_kill_background_task' to stop the task\n\n"
        result += f"Connection Details:\n"
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

        result = f"Background Task Status Report\n\n"
        result += f"Task ID: {task_id}\n"
        result += f"Process ID: {process_id}\n"
        result += f"Status: {'RUNNING' if process_status == 'RUNNING' else 'STOPPED/COMPLETED'}\n"
        result += f"Log Lines: {log_lines}\n"
        result += f"Host: {remote_host}:{remote_port}\n\n"

        if process_status == "RUNNING":
            result += f"Task is still running...\n\n"
        else:
            result += f"Task has completed or stopped\n\n"

        result += f"Recent Log Output (last {tail_lines} lines):\n"
        result += "=" * 50 + "\n"
        result += log_content
        result += "\n" + "=" * 50 + "\n\n"

        if process_status == "RUNNING":
            result += f"Task is still running. Check again later for updates."
        else:
            result += f"Task completed. Use 'ssh_execute_command' to clean up files if needed:\n"
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
            result = f"Task {task_id} (PID: {process_id}) is not running\n\n"
            result += "The process may have already completed or been killed.\n"
        else:
            kill_cmd = (
                f"kill {safe_pid_int} 2>/dev/null; sleep 2; "
                f"if kill -0 {safe_pid_int} 2>/dev/null; then kill -9 {safe_pid_int} 2>/dev/null; echo 'FORCE_KILLED'; else echo 'TERMINATED'; fi"
            )
            stdin, stdout, stderr = client.exec_command(kill_cmd)
            kill_result = stdout.read().decode("utf-8").strip()

            result = f"Background Task Killed\n\n"
            result += f"Task ID: {task_id}\n"
            result += f"Process ID: {process_id}\n"
            result += f"Kill Result: {kill_result}\n\n"

            if kill_result == "TERMINATED":
                result += f"Process terminated gracefully\n"
            elif kill_result == "FORCE_KILLED":
                result += f"Process force-killed (was unresponsive)\n"
            else:
                result += f"Unexpected result: {kill_result}\n"

        cleanup_cmd = (
            f"rm -f {safe_log} {safe_pid} 2>/dev/null; echo 'Cleanup attempted'"
        )
        stdin, stdout, stderr = client.exec_command(cleanup_cmd)
        cleanup_result = stdout.read().decode("utf-8").strip()

        result += f"\nCleanup: {cleanup_result}\n"
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

    In case connection error like "Error reading SSH protocol banner" - use port2 or direct instance ip,port instead
    """
    client = None
    sftp = None

    try:
        if not os.path.exists(local_file_path):
            return f"Error: Local file not found: {local_file_path}"

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

        try:
            remote_stat = sftp.stat(remote_file_path)
            remote_size = remote_stat.st_size
        except Exception:
            remote_size = "unknown"

        result = f"File Upload Successful!\n\n"
        result += f"Local File: {local_file_path}\n"
        result += f"Remote File: {remote_file_path}\n"
        result += f"Local Size: {local_size:,} bytes\n"
        if isinstance(remote_size, int):
            result += f"Remote Size: {remote_size:,} bytes\n"
        else:
            result += f"Remote Size: {remote_size}\n"
        result += f"Host: {remote_host}:{remote_port}\n"
        result += f"User: {remote_user}\n"

        if isinstance(remote_size, int) and local_size == remote_size:
            result += "\nFile transfer verified successfully!"
        elif isinstance(remote_size, int):
            result += (
                f"\nSize mismatch detected (local: {local_size}, remote: {remote_size})"
            )

        return result

    except FileNotFoundError:
        return f"Error: Local file not found: {local_file_path}"
    except paramiko.AuthenticationException:
        return f"Error: Authentication failed for {remote_user}@{remote_host}:{remote_port}"
    except paramiko.SSHException as e:
        return f"Error: SSH error occurred: {str(e)}"
    except Exception as e:
        return f"Error: Upload failed: {str(e)}"

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

        try:
            remote_stat = sftp.stat(remote_file_path)
            remote_size = remote_stat.st_size
        except FileNotFoundError:
            return f"Error: Remote file not found: {remote_file_path}"
        except Exception as e:
            return f"Error checking remote file: {str(e)}"

        logger.info(f"Downloading {remote_file_path} to {local_file_path}")

        sftp.get(remote_file_path, local_file_path)

        try:
            local_size = os.path.getsize(local_file_path)
        except Exception:
            local_size = "unknown"

        result = f"File Download Successful!\n\n"
        result += f"Remote File: {remote_file_path}\n"
        result += f"Local File: {local_file_path}\n"
        result += f"Remote Size: {remote_size:,} bytes\n"
        if isinstance(local_size, int):
            result += f"Local Size: {local_size:,} bytes\n"
        else:
            result += f"Local Size: {local_size}\n"
        result += f"Host: {remote_host}:{remote_port}\n"
        result += f"User: {remote_user}\n"

        if isinstance(local_size, int) and remote_size == local_size:
            result += "\nFile transfer verified successfully!"
        elif isinstance(local_size, int):
            result += (
                f"\nSize mismatch detected (remote: {remote_size}, local: {local_size})"
            )

        return result

    except paramiko.AuthenticationException:
        return f"Error: Authentication failed for {remote_user}@{remote_host}:{remote_port}"
    except paramiko.SSHException as e:
        return f"Error: SSH error occurred: {str(e)}"
    except Exception as e:
        return f"Error: Download failed: {str(e)}"

    finally:
        if sftp:
            sftp.close()
        if client:
            client.close()
        logger.info("SFTP and SSH connections closed")


@mcp.tool()
def show_ssh_keys(ctx: Context) -> str:
    """Show all SSH keys associated with the account"""
    try:
        response = vast.show_ssh_keys()

        if isinstance(response, dict):
            ssh_keys = response.get("ssh_keys", [])
        else:
            ssh_keys = response if isinstance(response, list) else []

        if not ssh_keys:
            return "No SSH keys found."

        result = f"SSH Keys ({len(ssh_keys)} found):\n\n"

        for key in ssh_keys:
            result += f"ID: {key.get('id', 'N/A')}\n"
            result += f"  Name: {key.get('name', 'N/A')}\n"
            result += f"  Fingerprint: {key.get('fingerprint', 'N/A')}\n"
            result += f"  Created: {key.get('created_at', 'N/A')}\n"
            result += "\n"

        return result

    except Exception as e:
        logger.error(f"Error showing SSH keys: {e}")
        return f"Error showing SSH keys: {str(e)}"


@mcp.tool()
def create_ssh_key(ctx: Context, ssh_key_str: str) -> str:
    """Create a new SSH key and add it to the account"""
    try:
        processed_ssh_key = get_ssh_key(ssh_key_str)

        response = vast.create_ssh_key(processed_ssh_key)

        if response.get("success") is True or response.get("id"):
            key_id = response.get("id", "unknown")
            return f"SSH key created successfully!\nKey ID: {key_id}"
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to create SSH key: {error_msg}"

    except ValueError as e:
        return f"Invalid SSH key: {str(e)}"
    except Exception as e:
        logger.error(f"Error creating SSH key: {e}")
        return f"Error creating SSH key: {str(e)}"


@mcp.tool()
def delete_ssh_key(ctx: Context, key_id: int) -> str:
    """Delete an SSH key from the account"""
    try:
        response = vast.delete_ssh_key(key_id)

        if response.get("success") is True:
            return f"SSH key {key_id} deleted successfully."
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to delete SSH key {key_id}: {error_msg}"

    except Exception as e:
        logger.error(f"Error deleting SSH key: {e}")
        return f"Error deleting SSH key {key_id}: {str(e)}"


@mcp.tool()
def detach_ssh(ctx: Context, instance_id: int, ssh_key_id: str) -> str:
    """Detach an SSH key from an instance"""
    try:
        response = vast.detach_ssh(instance_id, ssh_key_id)

        if response.get("success") is True:
            return f"SSH key {ssh_key_id} detached from instance {instance_id} successfully."
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to detach SSH key from instance {instance_id}: {error_msg}"

    except Exception as e:
        logger.error(f"Error detaching SSH key: {e}")
        return f"Error detaching SSH key from instance {instance_id}: {str(e)}"


@mcp.tool()
def show_volumes(ctx: Context, type: str = "all") -> str:
    """Show stats on owned volumes"""
    try:
        volumes = vast.show_volumes(type=type)

        if not volumes:
            return f"No {type} volumes found."

        result = f"Volumes ({len(volumes)} found):\n\n"

        for vol in volumes:
            result += f"ID: {vol.get('id', 'N/A')}\n"
            result += f"  Type: {vol.get('type', 'N/A')}\n"
            result += f"  Size: {vol.get('size', 0):.1f} GB\n"
            result += f"  Machine ID: {vol.get('machine_id', 'N/A')}\n"
            result += f"  Status: {vol.get('status', 'N/A')}\n"
            result += f"  Cost: ${vol.get('dph_total', 0):.4f}/hour\n"
            result += "\n"

        return result

    except Exception as e:
        logger.error(f"Error showing volumes: {e}")
        return f"Error showing volumes: {str(e)}"


@mcp.tool()
def create_volume(
    ctx: Context, offer_id: int, size: float = 15, name: str = None
) -> str:
    """Create a new volume from an offer"""
    try:
        kwargs = {"size": size}
        if name:
            kwargs["name"] = name

        response = vast.create_volume(offer_id, **kwargs)

        if response.get("success") is True:
            return f"Volume created successfully!\nOffer ID: {offer_id}\nSize: {size} GB\nName: {name or 'unnamed'}"
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to create volume: {error_msg}"

    except Exception as e:
        logger.error(f"Error creating volume: {e}")
        return f"Error creating volume: {str(e)}"


@mcp.tool()
def delete_volume(ctx: Context, volume_id: int) -> str:
    """Delete a volume"""
    try:
        response = vast.delete_volume(volume_id)

        if response.get("success") is True:
            return f"Volume {volume_id} deleted successfully."
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to delete volume {volume_id}: {error_msg}"

    except Exception as e:
        logger.error(f"Error deleting volume: {e}")
        return f"Error deleting volume {volume_id}: {str(e)}"


@mcp.tool()
def take_snapshot(ctx: Context, instance_id: int, repo: str = None) -> str:
    """Take a container snapshot and push to a registry"""
    try:
        kwargs = {}
        if repo:
            kwargs["repo"] = repo

        response = vast.take_snapshot(instance_id, **kwargs)

        if response.get("success") is True:
            return f"Snapshot of instance {instance_id} created successfully!"
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to take snapshot: {error_msg}"

    except Exception as e:
        logger.error(f"Error taking snapshot: {e}")
        return f"Error taking snapshot for instance {instance_id}: {str(e)}"


@mcp.tool()
def execute_command(ctx: Context, instance_id: int, command: str) -> str:
    """Execute a constrained command on a stopped instance"""
    try:
        result = vast.execute(instance_id, command)

        if isinstance(result, str):
            return (
                f"Command executed successfully on instance {instance_id}:\n\n{result}"
            )
        else:
            return f"Command executed on instance {instance_id}:\n\n{result}"

    except Exception as e:
        logger.error(f"Error executing command: {e}")
        return f"Error executing command on instance {instance_id}: {str(e)}"


@mcp.tool()
def update_instance(
    ctx: Context,
    instance_id: int,
    image: str = None,
    template_id: int = None,
    args: str = None,
    env: dict = None,
    onstart: str = None,
) -> str:
    """Update/recreate an instance from a new/updated template"""
    try:
        kwargs = {}
        if image is not None:
            kwargs["image"] = image
        if template_id is not None:
            kwargs["template_id"] = template_id
        if args is not None:
            kwargs["args"] = args
        if env is not None:
            kwargs["env"] = env
        if onstart is not None:
            kwargs["onstart"] = onstart

        response = vast.update_instance(instance_id, **kwargs)

        if response.get("success") is True:
            return f"Instance {instance_id} updated successfully."
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to update instance {instance_id}: {error_msg}"

    except Exception as e:
        logger.error(f"Error updating instance: {e}")
        return f"Error updating instance {instance_id}: {str(e)}"


@mcp.tool()
def change_bid(ctx: Context, instance_id: int, price: float) -> str:
    """Change the bid price for an instance"""
    try:
        response = vast.change_bid(instance_id, price)

        if response.get("success") is True:
            return f"Bid price for instance {instance_id} changed to ${price}/hour."
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to change bid for instance {instance_id}: {error_msg}"

    except Exception as e:
        logger.error(f"Error changing bid: {e}")
        return f"Error changing bid for instance {instance_id}: {str(e)}"


@mcp.tool()
def show_machines(ctx: Context) -> str:
    """Show all hosted machines"""
    try:
        machines = vast.show_machines()

        if not machines:
            return "No machines found."

        result = f"Machines ({len(machines)} found):\n\n"

        for machine in machines:
            result += f"ID: {machine.get('id', 'N/A')}\n"
            result += f"  Status: {machine.get('status', 'N/A')}\n"
            result += f"  GPU: {machine.get('gpu_name', 'N/A')}\n"
            result += f"  Total GPUs: {machine.get('total_gpus', 'N/A')}\n"
            result += f"  Available GPUs: {machine.get('num_gpus', 'N/A')}\n"
            result += f"  CPU: {machine.get('cpu_name', 'N/A')}\n"
            result += f"  RAM: {machine.get('cpu_ram', 0):.1f} GB\n"
            result += f"  Location: {machine.get('geolocation', 'N/A')}\n"
            result += "\n"

        return result

    except Exception as e:
        logger.error(f"Error showing machines: {e}")
        return f"Error showing machines: {str(e)}"


@mcp.tool()
def show_invoices(ctx: Context, start_date: str = None, end_date: str = None) -> str:
    """Show invoice/billing history"""
    try:
        kwargs = {}
        if start_date:
            kwargs["start_date"] = start_date
        if end_date:
            kwargs["end_date"] = end_date

        result_data = vast.show_invoices(**kwargs)

        if isinstance(result_data, dict):
            invoices = result_data.get("invoices", [])
            current = result_data.get("current", {})
        else:
            invoices = result_data if isinstance(result_data, list) else []
            current = {}

        result = f"Invoices ({len(invoices)} found):\n\n"

        for inv in invoices:
            result += f"ID: {inv.get('id', 'N/A')}\n"
            result += f"  Date: {inv.get('date', inv.get('when', 'N/A'))}\n"
            result += f"  Type: {inv.get('type', 'N/A')}\n"
            result += f"  Amount: ${inv.get('amount', 0):.2f}\n"
            if inv.get("description"):
                result += f"  Description: {inv.get('description')}\n"
            result += "\n"

        if current:
            result += f"\nCurrent Charges: ${current.get('total', 0):.2f}\n"

        return result

    except Exception as e:
        logger.error(f"Error showing invoices: {e}")
        return f"Error showing invoices: {str(e)}"


@mcp.tool()
def show_api_keys(ctx: Context) -> str:
    """Show all API keys"""
    try:
        response = vast.show_api_keys()

        if isinstance(response, dict):
            keys = response.get("api_keys", [])
        elif isinstance(response, list):
            keys = response
        else:
            keys = []

        if not keys:
            return "No API keys found."

        result = f"API Keys ({len(keys)} found):\n\n"

        for key in keys:
            result += f"ID: {key.get('id', 'N/A')}\n"
            result += f"  Name: {key.get('name', 'N/A')}\n"
            result += f"  Created: {key.get('created_at', 'N/A')}\n"
            if key.get("last_used"):
                result += f"  Last Used: {key.get('last_used')}\n"
            result += "\n"

        return result

    except Exception as e:
        logger.error(f"Error showing API keys: {e}")
        return f"Error showing API keys: {str(e)}"


@mcp.tool()
def create_api_key(ctx: Context, name: str, permissions: dict = None) -> str:
    """Create a new API key"""
    try:
        response = vast.create_api_key(name, permissions or {})

        if response.get("success") is True or response.get("id"):
            return f"API key created successfully!\nName: {name}\nID: {response.get('id', 'unknown')}"
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to create API key: {error_msg}"

    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        return f"Error creating API key: {str(e)}"


@mcp.tool()
def delete_api_key(ctx: Context, key_id: int) -> str:
    """Delete an API key"""
    try:
        response = vast.delete_api_key(key_id)

        if response.get("success") is True:
            return f"API key {key_id} deleted successfully."
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to delete API key {key_id}: {error_msg}"

    except Exception as e:
        logger.error(f"Error deleting API key: {e}")
        return f"Error deleting API key {key_id}: {str(e)}"


@mcp.tool()
def show_env_vars(ctx: Context) -> str:
    """Show environment variables (masked values)"""
    try:
        env_vars = vast.show_env_vars(show_values=False)

        if isinstance(env_vars, dict):
            items = list(env_vars.items())
        elif isinstance(env_vars, list):
            items = [
                (v.get("name", "unknown"), v.get("value", "****")) for v in env_vars
            ]
        else:
            items = []

        if not items:
            return "No environment variables found."

        result = f"Environment Variables ({len(items)} found):\n\n"

        for name, value in items:
            result += f"{name}: {value}\n"

        return result

    except Exception as e:
        logger.error(f"Error showing env vars: {e}")
        return f"Error showing environment variables: {str(e)}"


@mcp.tool()
def create_env_var(ctx: Context, name: str, value: str) -> str:
    """Create an environment variable"""
    try:
        response = vast.create_env_var(name, value)

        if response.get("success") is True:
            return f"Environment variable '{name}' created successfully."
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to create environment variable: {error_msg}"

    except Exception as e:
        logger.error(f"Error creating env var: {e}")
        return f"Error creating environment variable: {str(e)}"


@mcp.tool()
def update_env_var(ctx: Context, name: str, value: str) -> str:
    """Update an environment variable"""
    try:
        response = vast.update_env_var(name, value)

        if response.get("success") is True:
            return f"Environment variable '{name}' updated successfully."
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to update environment variable: {error_msg}"

    except Exception as e:
        logger.error(f"Error updating env var: {e}")
        return f"Error updating environment variable: {str(e)}"


@mcp.tool()
def delete_env_var(ctx: Context, name: str) -> str:
    """Delete an environment variable"""
    try:
        response = vast.delete_env_var(name)

        if response.get("success") is True:
            return f"Environment variable '{name}' deleted successfully."
        else:
            error_msg = response.get("msg", response.get("error", "Unknown error"))
            return f"Failed to delete environment variable: {error_msg}"

    except Exception as e:
        logger.error(f"Error deleting env var: {e}")
        return f"Error deleting environment variable: {str(e)}"


@mcp.tool()
def show_audit_logs(ctx: Context) -> str:
    """Show account audit logs"""
    try:
        logs = vast.show_audit_logs()

        if not logs:
            return "No audit logs found."

        result = f"Audit Logs ({len(logs)} entries):\n\n"

        for log in logs:
            result += f"ID: {log.get('id', 'N/A')}\n"
            result += f"  Time: {log.get('timestamp', log.get('created_at', 'N/A'))}\n"
            result += f"  Action: {log.get('action', 'N/A')}\n"
            result += (
                f"  Details: {log.get('details', log.get('description', 'N/A'))}\n"
            )
            result += "\n"

        return result

    except Exception as e:
        logger.error(f"Error showing audit logs: {e}")
        return f"Error showing audit logs: {str(e)}"


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
