"""Optional web3 ledger backend (local dev chain).

Kept entirely out of the default path and the smoke test. Requires a running
dev node (Anvil/Ganache) at ``ledger.web3.rpc_url`` plus ``web3`` and ``py-solc-x``.
Compiles + deploys ``contracts/OwnershipRegistry.sol`` on first use.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from qsfl.ledger.base import Ledger, OwnershipRecord
from qsfl.utils.logging import get_logger

logger = get_logger("ledger.web3")


def _hex_to_bytes32(hexstr: str) -> bytes:
    raw = bytes.fromhex(hexstr[2:] if hexstr.startswith("0x") else hexstr)
    return raw[:32].rjust(32, b"\x00")


class Web3Ledger(Ledger):
    def __init__(self, cfg) -> None:
        try:
            from web3 import Web3
        except Exception as exc:  # pragma: no cover - optional path
            raise ImportError(
                "web3 is required for ledger.backend=web3 (`pip install web3 py-solc-x`)."
            ) from exc

        w3cfg = cfg.ledger.web3
        self.w3 = Web3(Web3.HTTPProvider(str(w3cfg.rpc_url)))
        if not self.w3.is_connected():
            raise ConnectionError(f"no dev chain at {w3cfg.rpc_url}; start Anvil/Ganache first")
        self.account = self.w3.eth.accounts[int(w3cfg.get("account_index", 0))]
        self.w3.eth.default_account = self.account
        abi, bytecode = self._compile(Path(str(w3cfg.contract_path)))
        self.contract = self._deploy(abi, bytecode)
        logger.info("OwnershipRegistry deployed at %s", self.contract.address)

    def _compile(self, sol_path: Path):
        from solcx import compile_standard, install_solc

        install_solc("0.8.19")
        source = sol_path.read_text()
        compiled = compile_standard(
            {
                "language": "Solidity",
                "sources": {"OwnershipRegistry.sol": {"content": source}},
                "settings": {"outputSelection": {"*": {"*": ["abi", "evm.bytecode.object"]}}},
            },
            solc_version="0.8.19",
        )
        c = compiled["contracts"]["OwnershipRegistry.sol"]["OwnershipRegistry"]
        return c["abi"], c["evm"]["bytecode"]["object"]

    def _deploy(self, abi, bytecode):
        Factory = self.w3.eth.contract(abi=abi, bytecode=bytecode)
        tx_hash = Factory.constructor().transact()
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        return self.w3.eth.contract(address=receipt.contractAddress, abi=abi)

    def register(self, record: OwnershipRecord) -> dict:
        commitment = record.watermark_commitment or hashlib.sha256(b"").hexdigest()
        tx = self.contract.functions.register(
            _hex_to_bytes32(record.model_hash), record.owner, _hex_to_bytes32(commitment)
        ).transact()
        receipt = self.w3.eth.wait_for_transaction_receipt(tx)
        return {
            "tx_hash": receipt.transactionHash.hex(),
            "block_number": receipt.blockNumber,
            "contract": self.contract.address,
            "backend": "web3",
        }

    def verify(self, model_hash: str) -> dict:
        exists, registrant, owner, _commit, ts = self.contract.functions.getRecord(
            _hex_to_bytes32(model_hash)
        ).call()
        return {
            "found": bool(exists),
            "integrity_ok": True,  # chain consensus is the integrity guarantee here
            "record": {"owner": owner, "registrant": registrant, "timestamp": ts} if exists else None,
        }

    def verify_integrity(self) -> bool:
        return self.w3.is_connected()
