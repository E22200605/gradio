<script lang="ts">
	import MetaTags from "$lib/components/MetaTags.svelte";
	import { onMount, onDestroy } from "svelte";

	onMount(() => {
		document.body.style.overflow = "hidden";
	});

	onDestroy(() => {
		document.body.style.overflow = "";
	});
</script>

<MetaTags
	title="Workflow | Gradio"
	url="https://www.gradio.app/workflow"
	canonical="https://www.gradio.app/workflow"
	description="Gradio Workflow — build and run workflows visually."
/>

<main class="fixed inset-0 z-50 bg-white dark:bg-neutral-900">
	<iframe
		src="https://hmb-workflow.hf.space"
		title="Gradio Workflow"
		class="w-full h-full border-0 block"
		allow="accelerometer; camera; clipboard-read; clipboard-write; encrypted-media; fullscreen; geolocation; gyroscope; microphone; midi; payment"
		sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"
		loading="eager"
	></iframe>
</main>
