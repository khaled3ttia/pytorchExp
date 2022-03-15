

[external]
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 1) #3
3zextB+
)
	full_text

%11 = zext i32 %5 to i64
0addB)
'
	full_text

%12 = add i64 %10, %11
#i64B

	full_text
	
i64 %10
#i64B

	full_text
	
i64 %11
6truncB-
+
	full_text

%13 = trunc i64 %12 to i32
#i64B

	full_text
	
i64 %12
LcallBD
B
	full_text5
3
1%14 = tail call i64 @_Z13get_global_idj(i32 0) #3
3zextB+
)
	full_text

%15 = zext i32 %3 to i64
0addB)
'
	full_text

%16 = add i64 %14, %15
#i64B

	full_text
	
i64 %14
#i64B

	full_text
	
i64 %15
5icmpB-
+
	full_text

%17 = icmp slt i32 %13, %6
#i32B

	full_text
	
i32 %13
6truncB-
+
	full_text

%18 = trunc i64 %16 to i32
#i64B

	full_text
	
i64 %16
5icmpB-
+
	full_text

%19 = icmp slt i32 %18, %4
#i32B

	full_text
	
i32 %18
/andB(
&
	full_text

%20 = and i1 %17, %19
!i1B

	full_text


i1 %17
!i1B

	full_text


i1 %19
8brB2
0
	full_text#
!
br i1 %20, label %21, label %68
!i1B

	full_text


i1 %20
Ybitcast8BL
J
	full_text=
;
9%22 = bitcast double* %0 to [163 x [163 x [5 x double]]]*
Kbitcast8B>
<
	full_text/
-
+%23 = bitcast double* %1 to [164 x double]*
Kbitcast8B>
<
	full_text/
-
+%24 = bitcast double* %2 to [164 x double]*
5sext8B+
)
	full_text

%25 = sext i32 %7 to i64
1shl8B(
&
	full_text

%26 = shl i64 %12, 32
%i648B

	full_text
	
i64 %12
9ashr8B/
-
	full_text 

%27 = ashr exact i64 %26, 32
%i648B

	full_text
	
i64 %26
1shl8B(
&
	full_text

%28 = shl i64 %16, 32
%i648B

	full_text
	
i64 %16
9ashr8B/
-
	full_text 

%29 = ashr exact i64 %28, 32
%i648B

	full_text
	
i64 %28
¨getelementptr8B”
‘
	full_textƒ
€
~%30 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %22, i64 %25, i64 %27, i64 %29, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%31 = load double, double* %30, align 8, !tbaa !8
-double*8B

	full_text

double* %30
¨getelementptr8B”
‘
	full_textƒ
€
~%32 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %22, i64 %25, i64 %27, i64 %29, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%33 = load double, double* %32, align 8, !tbaa !8
-double*8B

	full_text

double* %32
¨getelementptr8B”
‘
	full_textƒ
€
~%34 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %22, i64 %25, i64 %27, i64 %29, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%35 = load double, double* %34, align 8, !tbaa !8
-double*8B

	full_text

double* %34
7fmul8B-
+
	full_text

%36 = fmul double %35, %35
+double8B

	full_text


double %35
+double8B

	full_text


double %35
icall8B_
]
	full_textP
N
L%37 = tail call double @llvm.fmuladd.f64(double %33, double %33, double %36)
+double8B

	full_text


double %33
+double8B

	full_text


double %33
+double8B

	full_text


double %36
¨getelementptr8B”
‘
	full_textƒ
€
~%38 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %22, i64 %25, i64 %27, i64 %29, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%39 = load double, double* %38, align 8, !tbaa !8
-double*8B

	full_text

double* %38
icall8B_
]
	full_textP
N
L%40 = tail call double @llvm.fmuladd.f64(double %39, double %39, double %37)
+double8B

	full_text


double %39
+double8B

	full_text


double %39
+double8B

	full_text


double %37
@fmul8B6
4
	full_text'
%
#%41 = fmul double %40, 5.000000e-01
+double8B

	full_text


double %40
¨getelementptr8B”
‘
	full_textƒ
€
~%42 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %22, i64 %25, i64 %27, i64 %29, i64 0
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%43 = load double, double* %42, align 8, !tbaa !8
-double*8B

	full_text

double* %42
7fdiv8B-
+
	full_text

%44 = fdiv double %41, %43
+double8B

	full_text


double %41
+double8B

	full_text


double %43
7fsub8B-
+
	full_text

%45 = fsub double %31, %44
+double8B

	full_text


double %31
+double8B

	full_text


double %44
@fmul8B6
4
	full_text'
%
#%46 = fmul double %45, 4.000000e-01
+double8B

	full_text


double %45
xgetelementptr8Be
c
	full_textV
T
R%47 = getelementptr inbounds [164 x double], [164 x double]* %23, i64 %27, i64 %29
=[164 x double]*8B&
$
	full_text

[164 x double]* %23
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nstore8BC
A
	full_text4
2
0store double %46, double* %47, align 8, !tbaa !8
+double8B

	full_text


double %46
-double*8B

	full_text

double* %47
4add8B+
)
	full_text

%48 = add nsw i32 %8, -1
6sext8B,
*
	full_text

%49 = sext i32 %48 to i64
%i328B

	full_text
	
i32 %48
¨getelementptr8B”
‘
	full_textƒ
€
~%50 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %22, i64 %49, i64 %27, i64 %29, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %49
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%51 = load double, double* %50, align 8, !tbaa !8
-double*8B

	full_text

double* %50
¨getelementptr8B”
‘
	full_textƒ
€
~%52 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %22, i64 %49, i64 %27, i64 %29, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %49
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%53 = load double, double* %52, align 8, !tbaa !8
-double*8B

	full_text

double* %52
¨getelementptr8B”
‘
	full_textƒ
€
~%54 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %22, i64 %49, i64 %27, i64 %29, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %49
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%55 = load double, double* %54, align 8, !tbaa !8
-double*8B

	full_text

double* %54
7fmul8B-
+
	full_text

%56 = fmul double %55, %55
+double8B

	full_text


double %55
+double8B

	full_text


double %55
icall8B_
]
	full_textP
N
L%57 = tail call double @llvm.fmuladd.f64(double %53, double %53, double %56)
+double8B

	full_text


double %53
+double8B

	full_text


double %53
+double8B

	full_text


double %56
¨getelementptr8B”
‘
	full_textƒ
€
~%58 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %22, i64 %49, i64 %27, i64 %29, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %49
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%59 = load double, double* %58, align 8, !tbaa !8
-double*8B

	full_text

double* %58
icall8B_
]
	full_textP
N
L%60 = tail call double @llvm.fmuladd.f64(double %59, double %59, double %57)
+double8B

	full_text


double %59
+double8B

	full_text


double %59
+double8B

	full_text


double %57
@fmul8B6
4
	full_text'
%
#%61 = fmul double %60, 5.000000e-01
+double8B

	full_text


double %60
¨getelementptr8B”
‘
	full_textƒ
€
~%62 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %22, i64 %49, i64 %27, i64 %29, i64 0
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %49
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nload8BD
B
	full_text5
3
1%63 = load double, double* %62, align 8, !tbaa !8
-double*8B

	full_text

double* %62
7fdiv8B-
+
	full_text

%64 = fdiv double %61, %63
+double8B

	full_text


double %61
+double8B

	full_text


double %63
7fsub8B-
+
	full_text

%65 = fsub double %51, %64
+double8B

	full_text


double %51
+double8B

	full_text


double %64
@fmul8B6
4
	full_text'
%
#%66 = fmul double %65, 4.000000e-01
+double8B

	full_text


double %65
xgetelementptr8Be
c
	full_textV
T
R%67 = getelementptr inbounds [164 x double], [164 x double]* %24, i64 %27, i64 %29
=[164 x double]*8B&
$
	full_text

[164 x double]* %24
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %29
Nstore8BC
A
	full_text4
2
0store double %66, double* %67, align 8, !tbaa !8
+double8B

	full_text


double %66
-double*8B

	full_text

double* %67
'br8B

	full_text

br label %68
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %8
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %4
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %7
$i328B

	full_text


i32 %5
,double*8B

	full_text


double* %2
$i328B

	full_text


i32 %6
,double*8B

	full_text


double* %1
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
4double8B&
$
	full_text

double 5.000000e-01
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 4
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 0
4double8B&
$
	full_text

double 4.000000e-01
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 1        		 
 
 

                    !    "# "" $% $& $' $( $$ )* )) +, +- +. +/ ++ 01 00 23 24 25 26 22 78 77 9: 9; 99 <= <> <? << @A @B @C @D @@ EF EE GH GI GJ GG KL KK MN MO MP MQ MM RS RR TU TV TT WX WY WW Z[ ZZ \] \^ \_ \\ `a `b `` cc de dd fg fh fi fj ff kl kk mn mo mp mq mm rs rr tu tv tw tx tt yz yy {| {} {{ ~ ~	€ ~	 ~~ ‚ƒ ‚
„ ‚
… ‚
† ‚‚ ‡ˆ ‡‡ ‰Š ‰
‹ ‰
Œ ‰‰ Ž   
‘ 
’ 
“  ”• ”” –— –
˜ –– ™š ™
› ™™ œ œœ žŸ ž
  ž
¡ žž ¢£ ¢
¤ ¢¢ ¥§ c¨ 	© ª 	« ¬ ­ 	® ¯     	  
       
 !  # % & '" ($ * , - ." /+ 1 3 4 5" 62 87 :7 ;0 =0 >9 ? A B C" D@ FE HE I< JG L N O P" QM SK UR V) XT YW [ ] ^" _Z a\ bc e gd h i" jf l nd o p" qm s ud v w" xt zy |y }r r €{  ƒd „ …" †‚ ˆ‡ Š‡ ‹~ Œ‰ Ž d ‘ ’" “ • —” ˜k š– ›™  Ÿ  " ¡œ £ž ¤  ¦¥ ¦ ±± °° ¦~ ±± ~G ±± G< ±± < °° ‰ ±± ‰ °° 	² K
² 	³ @
³ ‚	´ 2	´ tµ ¶ 	· $	· f	¸ 	¸ 	¸  	¸ "	¹ M
¹ 	º Z
º œ	» c	¼ +	¼ m"	
pintgr1"
_Z13get_global_idj"
llvm.fmuladd.f64*Š
npb-LU-pintgr1.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282

devmap_label
 

transfer_bytes	
Øº¶è

wgsize
@
 
transfer_bytes_log1p
	Œ£A

wgsize_log1p
	Œ£A