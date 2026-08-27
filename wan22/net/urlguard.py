from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

_METADATA_NETWORKS = (
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fe80::/10"),
)
_PRIVATE_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)


class UrlError(ValueError):
    """对外可映射成 400 的 URL 校验失败。"""


def host_allowed(host: str, allowlist: tuple[str, ...]) -> bool:
    host = host.lower().rstrip(".")
    if not host or not allowlist:
        return False
    for entry in allowlist:
        entry = entry.lower().rstrip(".")
        if not entry:
            continue
        if entry.startswith("."):
            suffix = entry
            bare = entry[1:]
            if host == bare or host.endswith(suffix):
                return True
        elif host == entry:
            return True
    return False


def assert_https_url(
    url: str,
    allowlist: tuple[str, ...],
    *,
    kind: str,
    allow_private: bool = False,
) -> str:
    """校验 https、白名单、无 userinfo。图片默认拒绝私网；回调允许内网但拒绝链路本地。"""
    parsed = urlparse(url.strip())
    if parsed.scheme != "https":
        raise UrlError(f"{kind} 只允许 https")
    if parsed.username or parsed.password:
        raise UrlError(f"{kind} 不能包含用户名密码")
    host = parsed.hostname
    if not host:
        raise UrlError(f"{kind} 缺少 host")
    if not host_allowed(host, allowlist):
        raise UrlError(f"{kind} host 不在白名单")
    _assert_resolved(host, allow_private=allow_private)
    return url.strip()


def assert_public_https(url: str, allowlist: tuple[str, ...], *, kind: str) -> str:
    return assert_https_url(url, allowlist, kind=kind, allow_private=False)


def _assert_resolved(host: str, *, allow_private: bool) -> None:
    addresses = _resolve(host)
    for ip in addresses:
        if any(ip in network for network in _METADATA_NETWORKS) or ip.is_link_local or ip.is_multicast:
            raise UrlError("URL 指向了不允许的地址")
        if allow_private:
            continue
        if ip.is_private or ip.is_loopback or any(ip in network for network in _PRIVATE_NETWORKS):
            raise UrlError("URL 指向了不允许的地址")


def _resolve(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        return [literal]
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UrlError("无法解析 URL host") from exc
    if not infos:
        raise UrlError("无法解析 URL host")
    return [ipaddress.ip_address(info[4][0]) for info in infos]
