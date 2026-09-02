"use client";

import { memo } from "react";
import { Handle, Position, type NodeProps } from "@xyflow/react";
import type { AgentNode as AgentNodeData } from "@/types/events";

/* Terminal-success agents keep their green dot; terminal-dead (stopped/crashed/
   failed) agents show a red dot. The name is dimmed + struck through for both. */
const DONE_DOT = "bg-emerald-500";
const DEAD_DOT = "bg-red-500";

/* Working agents pulse on a bezier curve. The authoritative "thinking" signal
   is the live llm_in_flight telemetry (fastest); executing a tool (inferred
   from the newest tool event) blinks medium; a running agent with neither is
   mid-transition between phases; waiting/parked blinks slowest. */
function blinkClass(
  status: string,
  runningTool: boolean | undefined,
  runtime: { llm_in_flight?: boolean } | null | undefined
): string | null {
  if (status === "running") {
    if (runtime?.llm_in_flight) return "strix-dot-fast";
    if (runningTool) return "strix-dot-medium";
    return "strix-dot-transition";
  }
  if (status === "waiting" || status === "budget_paused") return "strix-dot-slow";
  return null;
}

function isDead(status: string): boolean {
  return status === "stopped" || status === "crashed" || status === "failed" || status === "error";
}

function AgentNodeComponent({ data, selected }: NodeProps) {
  const agent = data as unknown as AgentNodeData & { isSelected: boolean };
  const isRoot = !agent.parentId;

  const done = agent.status === "completed";
  const dead = isDead(agent.status);
  const active = agent.status === "running" || agent.status === "waiting" || agent.status === "budget_paused";

  let dotClass: string;
  if (done) dotClass = DONE_DOT;
  else if (dead) dotClass = DEAD_DOT;
  else if (active && isRoot) dotClass = "bg-orange-500"; // unique root colour
  else if (active) dotClass = "bg-blue-500";
  else dotClass = "bg-gray-600";

  const blink = blinkClass(agent.status, agent.runningTool, agent.runtime);
  const terminal = done || dead;

  return (
    <div
      className={`w-[260px] rounded-lg border px-4 py-3 transition-colors ${
        agent.isSelected || selected
          ? "border-white/30 bg-[#0a0a0a]"
          : "border-[#222] bg-black hover:border-[#333]"
      }`}
    >
      <Handle
        type="target"
        position={Position.Top}
        isConnectable={false}
        className={`!w-1.5 !h-1.5 !border-0 ${agent.parentId ? "!bg-[#444]" : "!bg-transparent"}`}
      />

      <div className="flex items-center gap-2">
        <span
          className={`relative inline-flex h-2 w-2 shrink-0 rounded-full ${dotClass} ${
            blink ?? ""
          }`}
        />
        <span
          className={`text-sm font-semibold leading-snug line-clamp-3 transition-opacity ${
            terminal ? "opacity-60 line-through decoration-[#666] decoration-1" : "text-white"
          }`}
          title={isRoot ? `${agent.name} (orchestrator)` : agent.name}
        >
          {agent.name}
        </span>
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        isConnectable={false}
        className={`!w-1.5 !h-1.5 !border-0 ${
          agent.children && agent.children.length > 0 ? "!bg-[#444]" : "!bg-transparent"
        }`}
      />
    </div>
  );
}

export default memo(AgentNodeComponent);
