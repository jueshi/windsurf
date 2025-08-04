# Mermaid Diagram Syntax Guide

This guide provides standardized syntax rules for creating consistent and properly rendered Mermaid diagrams in markdown files.

## Key Syntax Rules

### 1. Node Labels
Always enclose node labels in double quotes:

```mermaid
graph LR
    A["Properly Quoted Node"] --> B["Another Node"]
```

❌ Incorrect:
```
A[Unquoted Node] --> B{Curly Braces Node}
```

✅ Correct:
```
A["Properly Quoted Node"] --> B["Square Bracket Node"]
```

### 2. Edge Text
Always enclose edge text in double quotes:

```mermaid
graph LR
    A["Start"] -->|"Edge with Text"| B["End"]
```

❌ Incorrect:
```
A["Start"] -->|Unquoted Edge Text| B["End"]
```

✅ Correct:
```
A["Start"] -->|"Properly Quoted Edge Text"| B["End"]
```

### 3. Subgraph Titles
Always enclose subgraph titles in double quotes:

```mermaid
graph TD
    subgraph "Process Flow"
        A["Step 1"] --> B["Step 2"]
    end
```

❌ Incorrect:
```
subgraph Process Flow
    A["Step 1"] --> B["Step 2"]
end
```

✅ Correct:
```
subgraph "Process Flow"
    A["Step 1"] --> B["Step 2"]
end
```

### 4. Node Definitions
Use square brackets `[]` consistently for node definitions:

```mermaid
graph LR
    A["Square Bracket Node"] --> B["Another Node"]
```

❌ Incorrect:
```
A{"Curly Brace Node"} --> B(("Double Parentheses Node"))
```

✅ Correct:
```
A["Properly Formatted Node"] --> B["Another Node"]
```

### 5. Style Consistency
Maintain consistent styling for nodes:

```mermaid
graph LR
    A["Node A"] --> B["Node B"]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style B fill:#bbf,stroke:#333,stroke-width:1px
```

### 6. Direction Statements
Keep direction statements within subgraphs:

```mermaid
graph TD
    subgraph "Group 1"
        direction LR
        A["First"] --> B["Second"]
    end
    
    subgraph "Group 2"
        direction RL
        C["Third"] --> D["Fourth"]
    end
```

## Complete Example

Here's a complete example incorporating all the syntax rules:

```mermaid
graph TD
    subgraph "Sender"
        direction LR
        Serializer --> CrispSignal["Crisp, Clear Signal"]
    end

    CrispSignal -->|"Travels Through..."| Channel["Communication Channel"]
    Channel -- "Introduces" --> Impairments["Loss, Distortion, ISI"]
    Impairments --> DistortedSignal["Weak, 'Blurry' Signal"]

    DistortedSignal --> Equalizer["Signal Equalizer"]
    Equalizer --> RestoredSignal["Sharpened, Clearer Signal"]

    RestoredSignal --> Deserializer["Receiver/Deserializer"]
    Deserializer --> InterpretedData["Interpreted Data"]

    style Impairments fill:#ffcccc
    style DistortedSignal fill:#ffdddd
    style Equalizer fill:#ccffcc
    style RestoredSignal fill:#ddffdd
```

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Diagram not rendering | Check for missing quotes around node labels, edge text, or subgraph titles |
| Inconsistent node shapes | Use square brackets `[]` consistently for all nodes |
| Text alignment problems | Ensure proper quoting and avoid special characters in unquoted text |
| Subgraph rendering issues | Always quote subgraph titles and check direction statements |

## Tools for Validation

- [Mermaid Live Editor](https://mermaid.live/): Test your diagrams before adding them to documentation
- VS Code extensions: "Markdown Preview Mermaid Support" and "Mermaid Markdown Syntax Highlighting"

Remember that consistent syntax across all diagrams improves readability and ensures proper rendering across different platforms and viewers.
