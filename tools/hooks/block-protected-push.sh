#!/usr/bin/env bash
# Blocks agents from pushing to the protected branches, development and master: changes go through
# a branch and a pull request. Used by Cursor (.cursor/hooks.json, beforeShellExecution) and by
# Claude Code (.claude/settings.json, PreToolUse on Bash). Each sends the command in its own JSON:
# Cursor as .command, Claude Code as .tool_input.command, and each reads the answer differently.
set -euo pipefail

input=$(cat)
command=$(jq -r '.command // .tool_input.command // empty' <<<"$input")
claude=$(jq -r 'has("tool_input")' <<<"$input")

protected='^(development|master)$'

pushes_to_protected() {
  # Each git push in the command, up to the next shell separator
  local segment refspecs=() token target
  while IFS= read -r segment; do
    read -r -a tokens <<<"$segment"
    refspecs=()
    for token in "${tokens[@]:2}"; do  # after "git push"
      [[ "$token" == -* ]] || refspecs+=("$token")
    done
    if ((${#refspecs[@]} <= 1)); then
      # git push [remote]: the current branch
      target=$(git symbolic-ref --short -q HEAD 2>/dev/null || true)
      [[ "$target" =~ $protected ]] && return 0
    fi
    for token in "${refspecs[@]:1}"; do  # after the remote
      target=${token#+}
      target=${target##*:}
      target=${target#refs/heads/}
      [[ "$target" =~ $protected ]] && return 0
    done
  done < <(grep -oE 'git[[:space:]]+push([^;&|]*)' <<<"$command" || true)
  return 1
}

if pushes_to_protected; then
  message="Direct pushes to development or master are blocked: push a chg/<issue>-<slug> or fix/<issue>-<slug> branch and open a pull request into development."
  if [[ "$claude" == true ]]; then
    echo "$message" >&2
    exit 2
  fi
  jq -n --arg message "$message" \
    '{permission: "deny", user_message: $message, agent_message: $message}'
  exit 0
fi

if [[ "$claude" != true ]]; then
  echo '{ "permission": "allow" }'
fi
exit 0
