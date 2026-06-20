// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/// @title OwnershipRegistry
/// @notice Minimal registry for model-ownership proofs. Stores H = SHA256(C)
///         (the encrypted-model hash) bound to an owner + watermark commitment,
///         and emits an event for off-chain indexing. Used by the optional
///         web3 ledger backend. A single-node dev chain is NOT a decentralized
///         trust anchor (see SECURITY.md).
contract OwnershipRegistry {
    struct Record {
        address registrant;
        string owner;
        bytes32 watermarkCommitment;
        uint256 timestamp;
        bool exists;
    }

    mapping(bytes32 => Record) private records;

    event Registered(
        bytes32 indexed modelHash,
        address indexed registrant,
        string owner,
        bytes32 watermarkCommitment,
        uint256 timestamp
    );

    /// @notice Register a model hash. Reverts if already registered (append-only).
    function register(bytes32 modelHash, string calldata owner, bytes32 watermarkCommitment) external {
        require(!records[modelHash].exists, "already registered");
        records[modelHash] = Record(msg.sender, owner, watermarkCommitment, block.timestamp, true);
        emit Registered(modelHash, msg.sender, owner, watermarkCommitment, block.timestamp);
    }

    /// @notice Look up a registered record.
    function getRecord(bytes32 modelHash)
        external
        view
        returns (bool exists, address registrant, string memory owner, bytes32 watermarkCommitment, uint256 timestamp)
    {
        Record storage r = records[modelHash];
        return (r.exists, r.registrant, r.owner, r.watermarkCommitment, r.timestamp);
    }
}
