package app

import (
	"encoding/json"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"
	"github.com/usestrix/strix/tui/internal/protocol"
)

func approval(requestID, action, reason string) *protocol.SafetyApproval {
	return &protocol.SafetyApproval{RequestID: requestID, Action: action, Reason: reason}
}

func TestSafetyApprovalPromptFollowsSnapshotAndDefaultsToDeny(t *testing.T) {
	model := New(nil)
	model.width, model.height = 130, 40
	model.ready = true
	model.showSplash = false

	model.handleEnvelope(stateEnvelope(t, 1, protocol.Snapshot{
		ScanState:       "running",
		PendingApproval: approval("approval-1", `{"cmd":"Run  exploit"}`, "This changes target state"),
	}))
	if model.modal != modalSafetyApproval || model.modalChoice != 1 {
		t.Fatalf("approval did not open fail-closed: modal=%v choice=%d", model.modal, model.modalChoice)
	}
	view := ansi.Strip(model.safetyApprovalView())
	for _, want := range []string{`Run  exploit`, "This changes target state", "Approve", "Deny"} {
		if !strings.Contains(view, want) {
			t.Fatalf("approval prompt is missing %q: %s", want, view)
		}
	}
	if rows := strings.Count(view, "\n") + 1; rows > 7 {
		t.Fatalf("approval prompt should stay compact, got %d rows:\n%s", rows, view)
	}

	// A newly dequeued request reuses the modal but must reset to Deny.
	model.modalChoice = 0
	model.handleEnvelope(stateEnvelope(t, 2, protocol.Snapshot{
		ScanState:       "running",
		PendingApproval: approval("approval-2", "Write file", "This changes the workspace"),
	}))
	if model.modal != modalSafetyApproval || model.modalChoice != 1 || model.safetyApprovalID != "approval-2" {
		t.Fatalf("next approval did not reset: modal=%v choice=%d id=%q", model.modal, model.modalChoice, model.safetyApprovalID)
	}

	model.handleEnvelope(stateEnvelope(t, 3, protocol.Snapshot{ScanState: "running"}))
	if model.modal != modalNone {
		t.Fatalf("cleared approval left modal open: %v", model.modal)
	}
}

func TestSafetyApprovalKeyboardSendsExactPayload(t *testing.T) {
	for _, tc := range []struct {
		name     string
		key      tea.KeyMsg
		choice   int
		approved bool
	}{
		{name: "approve selected", key: tea.KeyMsg{Type: tea.KeyEnter}, choice: 0, approved: true},
		{name: "deny default", key: tea.KeyMsg{Type: tea.KeyEnter}, choice: 1, approved: false},
		{name: "escape denies", key: tea.KeyMsg{Type: tea.KeyEsc}, choice: 0, approved: false},
		{name: "approve shortcut", key: tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'a'}}, choice: 1, approved: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			connection := &recordingConn{}
			model := New(&Client{conn: connection})
			model.width, model.height = 130, 40
			model.snapshot.PendingApproval = approval("approval-exact", "Action", "Reason")
			model.openModal(modalSafetyApproval)
			model.modalChoice = tc.choice

			updated, cmd := model.updateModal(tc.key)
			model = updated.(Model)
			envelope := commandFromCmd(t, cmd, connection)
			if envelope.Type != "safety.resolve" {
				t.Fatalf("command = %q, want safety.resolve", envelope.Type)
			}
			var payload struct {
				RequestID string `json:"request_id"`
				Approved  bool   `json:"approved"`
			}
			if err := json.Unmarshal(envelope.Payload, &payload); err != nil {
				t.Fatal(err)
			}
			if payload.RequestID != "approval-exact" || payload.Approved != tc.approved {
				t.Fatalf("payload = %#v, want id=%q approved=%v", payload, "approval-exact", tc.approved)
			}
			if model.modal != modalSafetyApproval {
				t.Fatalf("approval closed before backend state cleared it: %v", model.modal)
			}
		})
	}
}

func TestSafetyApprovalMouseButtonsSendPayload(t *testing.T) {
	for _, tc := range []struct {
		label    string
		approved bool
	}{
		{label: "Approve", approved: true},
		{label: "Deny", approved: false},
	} {
		t.Run(tc.label, func(t *testing.T) {
			connection := &recordingConn{}
			model := New(&Client{conn: connection})
			model.width, model.height = 130, 40
			model.ready = true
			model.snapshot.PendingApproval = approval("approval-mouse", "Action", "Reason")
			model.openModal(modalSafetyApproval)
			view := model.modalView()
			left, top, _, _ := model.cornerViewBounds(view)
			x, y := -1, -1
			for row, line := range strings.Split(view, "\n") {
				plain := ansi.Strip(line)
				if index := strings.Index(plain, tc.label); index >= 0 {
					x = left + ansi.StringWidth(plain[:index])
					y = top + row
					break
				}
			}
			if x < 0 {
				t.Fatalf("button %q was not rendered", tc.label)
			}

			updated, cmd := model.updateModalMouse(tea.MouseMsg{
				X: x, Y: y, Button: tea.MouseButtonLeft, Action: tea.MouseActionPress,
			})
			model = updated.(Model)
			envelope := commandFromCmd(t, cmd, connection)
			var payload struct {
				RequestID string `json:"request_id"`
				Approved  bool   `json:"approved"`
			}
			if err := json.Unmarshal(envelope.Payload, &payload); err != nil {
				t.Fatal(err)
			}
			if payload.RequestID != "approval-mouse" || payload.Approved != tc.approved {
				t.Fatalf("payload = %#v", payload)
			}
		})
	}
}

func TestSafetyApprovalDoesNotTrapQuitKeys(t *testing.T) {
	for _, key := range []tea.KeyMsg{
		{Type: tea.KeyCtrlC},
		{Type: tea.KeyCtrlQ},
	} {
		connection := &recordingConn{}
		model := New(&Client{conn: connection})
		model.snapshot.PendingApproval = approval("approval-quit", "Action", "Reason")
		model.openModal(modalSafetyApproval)

		updated, _ := model.updateModal(key)
		model = updated.(Model)
		if model.modal != modalQuit || model.modalChoice != 1 {
			t.Fatalf("quit key did not open fail-closed quit confirmation: modal=%v choice=%d", model.modal, model.modalChoice)
		}
		model.handleEnvelope(stateEnvelope(t, 1, protocol.Snapshot{
			ScanState:       "running",
			PendingApproval: approval("approval-quit", "Action", "Reason"),
		}))
		if model.modal != modalQuit {
			t.Fatalf("state refresh displaced quit confirmation: modal=%v", model.modal)
		}

		// Declining quit must restore the still-pending approval.
		updated, _ = model.updateModal(tea.KeyMsg{Type: tea.KeyEnter})
		model = updated.(Model)
		if model.modal != modalSafetyApproval || model.modalChoice != 1 {
			t.Fatalf("declining quit did not restore approval: modal=%v choice=%d", model.modal, model.modalChoice)
		}
	}
}

func TestQueuedSafetyResolutionsUseDistinctPendingKeys(t *testing.T) {
	first := pendingKey("safety.resolve", json.RawMessage(`{"request_id":"approval-1","approved":true}`))
	opposite := pendingKey("safety.resolve", json.RawMessage(`{"request_id":"approval-1","approved":false}`))
	second := pendingKey("safety.resolve", json.RawMessage(`{"request_id":"approval-2","approved":true}`))
	if first != opposite {
		t.Fatal("opposite answers for one safety request use different pending keys")
	}
	if first == second {
		t.Fatal("queued safety resolutions share one pending command key")
	}
}

func TestSafetyApprovalDisablesApproveWhenExactContentDoesNotFit(t *testing.T) {
	connection := &recordingConn{}
	model := New(&Client{conn: connection})
	model.width, model.height = 32, 10
	model.snapshot.PendingApproval = approval("approval-small", strings.Repeat("x", 300), strings.Repeat("reason ", 20))
	model.openModal(modalSafetyApproval)
	model.modalChoice = 0

	if model.safetyApprovalFits() {
		t.Fatal("oversized approval unexpectedly fits the terminal")
	}
	if view := ansi.Strip(model.safetyApprovalView()); !strings.Contains(view, "Approval is disabled") {
		t.Fatalf("small-terminal warning missing: %s", view)
	}
	updated, cmd := model.updateModal(tea.KeyMsg{Type: tea.KeyEnter})
	model = updated.(Model)
	if cmd != nil {
		t.Fatal("approval command was sent without displaying exact content")
	}
	if !strings.Contains(model.errorText, "Resize the terminal") {
		t.Fatalf("missing resize guidance: %q", model.errorText)
	}
}
