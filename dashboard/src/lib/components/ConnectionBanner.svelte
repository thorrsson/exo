<script lang="ts">
  import { getClusterHealth, isConnected } from "$lib/stores/app.svelte";
  import { slide } from "svelte/transition";

  const connected = $derived(isConnected());
  const health = $derived(getClusterHealth());
</script>

{#if !connected}
  <div
    transition:slide={{ duration: 200 }}
    class="relative z-50 flex items-center justify-center gap-2 px-4 py-2 bg-red-950/80 border-b border-red-500/30"
    role="alert"
    aria-live="assertive"
  >
    <div class="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
    <span class="text-xs font-mono text-red-300 tracking-wider uppercase">
      Connection lost &mdash; Reconnecting to backend&hellip;
    </span>
  </div>
{:else if health?.status !== "healthy"}
  <div
    transition:slide={{ duration: 200 }}
    class="relative z-50 flex flex-wrap items-center justify-center gap-2 px-4 py-2 bg-amber-950/80 border-b border-amber-500/30"
    role="status"
    aria-live="polite"
  >
    <div class="w-2 h-2 bg-amber-500 rounded-full animate-pulse"></div>
    <span class="text-xs font-mono text-amber-200">
      {#if health}
        Cluster {health.status} &mdash; {health.issues.join(". ")}.
        {#if health.unavailableNodeIds.length}
          Peers: {health.unavailableNodeIds.map((id) => id.slice(0, 8)).join(", ")}.
        {/if}
        {#if health.retryInSeconds > 0}
          Election retry available in {Math.ceil(health.retryInSeconds)}s.
        {/if}
      {:else}
        Cluster health unavailable &mdash; checking backend&hellip;
      {/if}
    </span>
  </div>
{/if}
