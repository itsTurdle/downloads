"""Local TLS certificates, because iOS will not give a web page the camera over
plain HTTP.

`getUserMedia` requires a secure context, and `http://192.168.x.x:8770` is not one --
Safari does not even show a permission prompt, it just fails. So the capture page has
to be served over HTTPS, which on a LAN means minting our own certificate.

Two ways to make the phone accept it:

  1. Open the page and tap through Safari's warning ("Show Details" -> "visit this
     website"). Quickest, and enough for the camera prompt to appear.
  2. Install `ca.crt` once (the bridge serves it at /ca.crt) and enable it under
     Settings -> General -> About -> Certificate Trust Settings. Then no warnings at
     all, on any of these pages, until it expires.

A CA plus a leaf rather than one self-signed cert, so option 2 exists. Certificates
land next to this file and are reused; the leaf is reissued when the address list
changes, since iOS matches on subjectAltName and ignores commonName entirely.
"""

import datetime
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

CERT_DIR = Path(__file__).resolve().parent / "_certs"

CA_CERT = CERT_DIR / "ca.crt"
CA_KEY = CERT_DIR / "ca.key"
LEAF_CERT = CERT_DIR / "server.crt"
LEAF_KEY = CERT_DIR / "server.key"
SANS_STAMP = CERT_DIR / "sans.txt"

# iOS rejects server certificates valid for more than 825 days, and in practice
# anything over ~398. Stay well inside that.
LEAF_DAYS = 365
CA_DAYS = 3650


def _now():
    return datetime.datetime.now(datetime.timezone.utc)


def _write_private(path: Path, key):
    path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ))


def _load_or_make_ca():
    if CA_CERT.exists() and CA_KEY.exists():
        key = serialization.load_pem_private_key(CA_KEY.read_bytes(), password=None)
        cert = x509.load_pem_x509_certificate(CA_CERT.read_bytes())
        if cert.not_valid_after_utc > _now():
            return cert, key

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "iPhone LiDAR local CA"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "iPhone LiDAR bridge"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now() - datetime.timedelta(days=1))
        .not_valid_after(_now() + datetime.timedelta(days=CA_DAYS))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.KeyUsage(
            digital_signature=True, content_commitment=False, key_encipherment=False,
            data_encipherment=False, key_agreement=False, key_cert_sign=True,
            crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
        .sign(key, hashes.SHA256())
    )
    CERT_DIR.mkdir(exist_ok=True)
    CA_CERT.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    _write_private(CA_KEY, key)
    return cert, key


def _san_entries(hosts: list[str]):
    entries = []
    for h in hosts:
        try:
            entries.append(x509.IPAddress(ipaddress.ip_address(h)))
        except ValueError:
            entries.append(x509.DNSName(h))
    return entries


def ensure_cert(hosts: list[str]) -> tuple[Path, Path, Path]:
    """Return (leaf_cert, leaf_key, ca_cert), issuing them if needed."""
    CERT_DIR.mkdir(exist_ok=True)
    hosts = sorted(set(hosts + ["localhost", "127.0.0.1"]))
    stamp = "\n".join(hosts)

    fresh = (
        LEAF_CERT.exists() and LEAF_KEY.exists()
        and SANS_STAMP.exists() and SANS_STAMP.read_text() == stamp
    )
    if fresh:
        cert = x509.load_pem_x509_certificate(LEAF_CERT.read_bytes())
        if cert.not_valid_after_utc > _now() + datetime.timedelta(days=1):
            return LEAF_CERT, LEAF_KEY, CA_CERT

    ca_cert, ca_key = _load_or_make_ca()
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "iphone-lidar bridge"),
        ]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now() - datetime.timedelta(days=1))
        .not_valid_after(_now() + datetime.timedelta(days=LEAF_DAYS))
        # iOS ignores commonName outright; only these matter.
        .add_extension(x509.SubjectAlternativeName(_san_entries(hosts)), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(x509.ExtendedKeyUsage([x509.ObjectIdentifier("1.3.6.1.5.5.7.3.1")]),
                       critical=False)   # serverAuth, which iOS requires
        .sign(ca_key, hashes.SHA256())
    )
    LEAF_CERT.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    _write_private(LEAF_KEY, key)
    SANS_STAMP.write_text(stamp)
    return LEAF_CERT, LEAF_KEY, CA_CERT
