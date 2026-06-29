package com.vulnseer.util;

import org.objectweb.asm.Type;

import java.util.Arrays;
import java.util.stream.Collectors;

public class MethodUtil {

    public static String getMethodSignature(String className, String methodName, Type returnType, Type[] argTypes) {
        return "<" +
                className +
                ": " +
                corp(returnType.getClassName()) +
                " " +
                methodName +
                "(" +
                Arrays.stream(argTypes)
                        .map(org.objectweb.asm.Type::getClassName)
                        .map(MethodUtil::corp)
                        .collect(Collectors.joining(",")) +
                ")" +
                ">";
    }

    private static String corp(String name) {
        if (name.charAt(0) == '[') {
            int j = 0;
            int cnt = 0;
            while (name.charAt(j) == '[') {
                j++;
                cnt++;
            }

            if (name.charAt(j) == 'L') j++;

            name = name.substring(j);

            if (name.charAt(name.length() - 1) == ';') {
                name = name.substring(0, name.length() - 1);
            }

            switch (name) {
                case "V":
                    name = "void";
                    break;
                case "Z":
                    name = "boolean";
                    break;
                case "B":
                    name = "byte";
                    break;
                case "C":
                    name = "char";
                    break;
                case "S":
                    name = "short";
                    break;
                case "I":
                    name = "int";
                    break;
                case "J":
                    name = "long";
                    break;
                case "F":
                    name = "float";
                    break;
                case "D":
                    name = "double";
                    break;
                default:
                    break;
            }

            StringBuilder sb = new StringBuilder(name);
            for (int i = 0; i < cnt; i++) sb.append("[]");
            name = sb.toString();
            return name;
        } else {
            return name;
        }
    }
}
