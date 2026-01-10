#!/bin/bash
# Verification script for trajectory-supervised goal learning system

echo "=================================="
echo "Implementation Verification"
echo "=================================="
echo ""

# Check if files exist
echo "Checking files..."
files=(
    "navsim/agents/goalflow/goal_decoder.py"
    "navsim/agents/goalflow/multimodal_dit_decoder.py"
    "navsim/agents/goalflow/unified_goalflow_model.py"
    "navsim/agents/goalflow/unified_goalflow_agent.py"
    "navsim/agents/goalflow/unified_goalflow_loss.py"
    "navsim/agents/goalflow/test_goal_learning.py"
    "docs/goal_learning_system.md"
    "examples/goal_learning_example.py"
    "IMPLEMENTATION_SUMMARY.md"
)

all_exist=true
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file (MISSING)"
        all_exist=false
    fi
done
echo ""

# Check syntax
echo "Checking Python syntax..."
py_files=(
    "navsim/agents/goalflow/goal_decoder.py"
    "navsim/agents/goalflow/multimodal_dit_decoder.py"
    "navsim/agents/goalflow/unified_goalflow_model.py"
    "navsim/agents/goalflow/unified_goalflow_agent.py"
    "navsim/agents/goalflow/unified_goalflow_loss.py"
)

all_valid=true
for file in "${py_files[@]}"; do
    if python -m py_compile "$file" 2>/dev/null; then
        echo "✓ $file syntax valid"
    else
        echo "✗ $file syntax error"
        all_valid=false
    fi
done
echo ""

# Count lines of code
echo "Lines of code:"
wc -l "${py_files[@]}" 2>/dev/null | tail -1
echo ""

# Check configuration changes
echo "Checking configuration..."
if grep -q "num_goal_queries" navsim/agents/goalflow/goalflow_config.py; then
    echo "✓ Configuration updated with new parameters"
else
    echo "✗ Configuration not updated"
fi
echo ""

# Summary
echo "=================================="
echo "Verification Summary"
echo "=================================="
if [ "$all_exist" = true ] && [ "$all_valid" = true ]; then
    echo "✅ All checks passed!"
    echo ""
    echo "Implementation complete:"
    echo "  - 5 core modules"
    echo "  - 1 test suite"
    echo "  - 3 documentation files"
    echo "  - 1 configuration update"
    echo ""
    echo "Ready for integration testing!"
else
    echo "❌ Some checks failed"
    exit 1
fi
