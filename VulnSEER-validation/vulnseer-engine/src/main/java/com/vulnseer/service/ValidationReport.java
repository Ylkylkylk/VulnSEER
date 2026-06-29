package com.vulnseer.service;

import lombok.Data;

import java.util.List;
import java.util.Map;

@Data
public class ValidationReport {

    private String cveName;
    private String chainId;
    private List<String> callChain;
    private String entryMethod;
    private String sinkMethod;

    private String resultStatus;
    private boolean sinkReached;
    private boolean vulnerabilityTriggered;

    private Long analysisTime;
    private Long fuzzTime;
    private Long totalTime;

    private Integer successStep;
    private Long fuzzChainTime;

    private List<Map<String, String>> argsMapList;
    private Map<String, String> fieldMap;
    private Map<String, String> invokeResultMap;

    private Integer numDecisions;
    private Integer numInclude;
    private Integer numExclude;
    private Boolean truncated;
    private Double elapsedSec;

    private String description;
    private String notes;

    private Map<String, Object> rawInput;
}