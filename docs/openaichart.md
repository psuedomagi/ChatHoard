
%%{init: {"flowchart": {"htmlLabels": false}} }%%
%%{init: {"flowchart": {"defaultRenderer": "elk"}} }%%

graph TD
    %% Main structure
    A[ConversationHistoryJson] --> B[Conversation]
    B --> C[ChatSession]
    C --> D[Message]

    %% Message attributes
    subgraph Message_Attributes
        D --> E1[MessageAuthor]
        D --> E2[MessageContent]
        D --> E3[MessageMetadata]
    end

    %% Discriminated Unions
    subgraph Discriminated_Unions
        E2 --> |"Discriminated by content_type"| E3
    end

    %% MessageContent and its subgroups
    subgraph MessageContent
        E2 --> F1[MessageTextContent]
        E2 --> F3[MessageCodeContent]
        E2 --> F5[MessageTetherBrowsingDisplayContent]
        E2 --> F7[MessageTetherQuoteContent]
        E2 --> F9[MessageExecutionOutputContent]
        E2 --> F11[MessageSystemErrorContent]
        E2 --> F13[MessageMultimodalTextContent]

        F13 --> G1[MessageContentImagePart]
        G1 --> H1[MessageContentImagePartMetadata]
        H1 --> I1[DalleMetadata]
    end

    %% MessageMetadata and its subgroups
    subgraph MessageMetadata
        E3 --> F2[MessageTextMetadata]
        E3 --> F4[MessageCodeMetadata]
        E3 --> F6[MessageTetherBrowsingDisplayMetadata]
        E3 --> F8[MessageTetherQuoteMetadata]
        E3 --> F10[MessageExecutionOutputMetadata]
        E3 --> F12[MessageSystemErrorMetadata]
        E3 --> F14[MessageMultimodalTextMetadata]
    end

    %% Content and Metadata relationships
    F1 -.-> F2
    F3 -.-> F4
    F5 -.-> F6
    F7 -.-> F8
    F9 -.-> F10
    F11 -.-> F12
    F13 -.-> F14

    %% Metadata types to top-level metadata attributes
    subgraph Attachments
        F2 --> G2[Attachments]
        F4 --> G2
        F14 --> G2
    end

    subgraph FinishDetails
        F2 --> G3[FinishDetails]
        F4 --> G3
        F10 --> G3
    end

    subgraph UserContextMessageData
        F2 --> G4[UserContextMessageData]
    end

    subgraph JitPluginData
        F2 --> G5[JitPluginData]
        G5 --> H5[JitFromServer]
        H5 --> I5[JitFromServerBody]
        I5 --> J2[JitFromServerBodyActions]
        J2 --> K1[JitFromServerBodyActionsTarget]
        I5 --> J3[JitFromServerBodyParams]
        G5 --> H6[JitFromClient]
        H6 --> I6[JitFromClientTarget]
        I6 --> J4[JitFromClientTargetData]
    end

    subgraph Citations
        F2 --> G6[Citations]
        F4 --> G6
        G6 --> H4[CitationMetadata]
        H4 --> I4[CitationMetadataExtra]
    end

    subgraph CiteMetadata
        F6 --> G7[CiteMetadata]
        F8 --> G7
        G7 --> I7[InsideCiteMetadataList]
        I7 --> J5[CiteMetadataCitationFormat]
    end

    subgraph AggregateResult
        F10 --> G8[AggregateResult]
        G8 --> H2[InKernelException]
        G8 --> H3[AggregateResultJupyterMessage]
        H3 --> I2[AggregateResultJupyterMessageParentHeader]
        H3 --> I3[AggregateResultJupyterMessageContent]
        I3 --> J1[AggregateResultJupyterMessageContentData]
    end
