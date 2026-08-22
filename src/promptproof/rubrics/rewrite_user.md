# Failing instruction

<instruction id="$instruction_id">
$instruction_text
</instruction>

# Input that defeated it

<input axis="$axis" technique="$strategy">
$failing_input
</input>

# What the assistant produced

<evidence>
$evidence
</evidence>

# Judge's reasoning

<judge_reasoning>
$reasoning
</judge_reasoning>

Return the JSON object with the replacement instruction.
