package protocol

import (
	"encoding/json"
	"reflect"
	"testing"
)

func TestProtocolVersionAndCapabilities(t *testing.T) {
	if Version != 4 {
		t.Fatalf("protocol version = %d, want 4", Version)
	}
	wantCapabilities := []string{
		"state-revisions",
		"collection-deltas",
		"structured-command-errors",
		"agents-collection",
		"safety-approval",
	}
	if !reflect.DeepEqual(Capabilities, wantCapabilities) {
		t.Fatalf("capabilities = %#v, want %#v", Capabilities, wantCapabilities)
	}

}

func TestSnapshotDecodesPendingSafetyApproval(t *testing.T) {
	var snapshot Snapshot
	if err := json.Unmarshal([]byte(`{
		"pending_approval": {
			"request_id": "approval-1",
			"action": "Run exploit",
			"reason": "Changes target state"
		}
	}`), &snapshot); err != nil {
		t.Fatal(err)
	}
	if snapshot.PendingApproval == nil {
		t.Fatal("pending approval was not decoded")
	}
	if got := *snapshot.PendingApproval; got != (SafetyApproval{
		RequestID: "approval-1",
		Action:    "Run exploit",
		Reason:    "Changes target state",
	}) {
		t.Fatalf("pending approval = %#v", got)
	}
}
