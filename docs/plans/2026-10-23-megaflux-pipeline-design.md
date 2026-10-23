# SM90 W1/W2 pipeline implementation design

Implement after accepting and committing GPU planning. Keep the committed
Plan unchanged between synchronous and pipelined comparison.

1. Preserve symmetric complete-leg exports and the fixed physical bank.
   Publish all source exports before starting asynchronous receiver streams.
2. Use separate W1 and W2 CUDA streams, exact-byte gets and quiet on each
   stream, then unpack. Publish epoch-qualified leg/slot ready flags only after
   complete payload, folded offset and residual stores plus a system fence.
3. Gate weight TMA/offset producers and residual consumers with acquire loads
   of the appropriate leg's ready flag. Issue the async-proxy fence before TMA.
4. The original hybrid epilogue eagerly reads both legs' residuals even on
   FC1. Maintain an attributed copy of its task loop that reads only the
   current phase's residual, preserving arithmetic and all other operations.
5. Reserve persistent CTA capacity for independent communication progress.
   Derive dispatch/grid barrier counts from the reduced launch cluster count;
   do not merely reduce a host scheduler hint.
6. Retire both communication streams and old compute before repacking exports
   or reusing any slot. Keep capture disabled until the separate Graph stage.
7. Inject W1, W2 payload, folded-offset and residual delays. Compare outputs
   and full bytes to synchronous execution of the same GPU Plan. Device
   telemetry must prove FC1 completion while W2 remains delayed, and replica
   FC1/FC2 consumption must follow their respective ready publications.
8. Run EP2 first and EP8 across changing plans, masked and ragged sources,
   tails and repeated slot reuse. Record resident CTA/resource changes and
   test logs, then commit this accepted stage independently.
