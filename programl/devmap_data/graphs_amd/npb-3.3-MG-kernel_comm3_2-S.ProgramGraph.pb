

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 0) #2
4truncB+
)
	full_text

%7 = trunc i64 %6 to i32
"i64B

	full_text


i64 %6
3icmpB+
)
	full_text

%8 = icmp slt i32 %7, %1
"i32B

	full_text


i32 %7
6brB0
.
	full_text!

br i1 %8, label %9, label %44
 i1B

	full_text	

i1 %8
Ncall8BD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 1) #2
0add8B'
%
	full_text

%11 = add i64 %10, 1
%i648B

	full_text
	
i64 %10
8trunc8B-
+
	full_text

%12 = trunc i64 %11 to i32
%i648B

	full_text
	
i64 %11
0mul8B'
%
	full_text

%13 = mul i32 %2, %1
2mul8B)
'
	full_text

%14 = mul i32 %13, %12
%i328B

	full_text
	
i32 %13
%i328B

	full_text
	
i32 %12
4add8B+
)
	full_text

%15 = add nsw i32 %2, -2
5mul8B,
*
	full_text

%16 = mul nsw i32 %15, %1
%i328B

	full_text
	
i32 %15
1add8B(
&
	full_text

%17 = add i32 %16, %4
%i328B

	full_text
	
i32 %16
1add8B(
&
	full_text

%18 = add i32 %17, %7
%i328B

	full_text
	
i32 %17
$i328B

	full_text


i32 %7
2add8B)
'
	full_text

%19 = add i32 %18, %14
%i328B

	full_text
	
i32 %18
%i328B

	full_text
	
i32 %14
6sext8B,
*
	full_text

%20 = sext i32 %19 to i64
%i328B

	full_text
	
i32 %19
^getelementptr8BK
I
	full_text<
:
8%21 = getelementptr inbounds double, double* %0, i64 %20
%i648B

	full_text
	
i64 %20
Abitcast8B4
2
	full_text%
#
!%22 = bitcast double* %21 to i64*
-double*8B

	full_text

double* %21
Hload8B>
<
	full_text/
-
+%23 = load i64, i64* %22, align 8, !tbaa !8
'i64*8B

	full_text


i64* %22
0add8B'
%
	full_text

%24 = add i32 %7, %4
$i328B

	full_text


i32 %7
2add8B)
'
	full_text

%25 = add i32 %24, %14
%i328B

	full_text
	
i32 %24
%i328B

	full_text
	
i32 %14
6sext8B,
*
	full_text

%26 = sext i32 %25 to i64
%i328B

	full_text
	
i32 %25
^getelementptr8BK
I
	full_text<
:
8%27 = getelementptr inbounds double, double* %0, i64 %26
%i648B

	full_text
	
i64 %26
Abitcast8B4
2
	full_text%
#
!%28 = bitcast double* %27 to i64*
-double*8B

	full_text

double* %27
Hstore8B=
;
	full_text.
,
*store i64 %23, i64* %28, align 8, !tbaa !8
%i648B

	full_text
	
i64 %23
'i64*8B

	full_text


i64* %28
0add8B'
%
	full_text

%29 = add i32 %4, %1
1add8B(
&
	full_text

%30 = add i32 %29, %7
%i328B

	full_text
	
i32 %29
$i328B

	full_text


i32 %7
2add8B)
'
	full_text

%31 = add i32 %30, %14
%i328B

	full_text
	
i32 %30
%i328B

	full_text
	
i32 %14
6sext8B,
*
	full_text

%32 = sext i32 %31 to i64
%i328B

	full_text
	
i32 %31
^getelementptr8BK
I
	full_text<
:
8%33 = getelementptr inbounds double, double* %0, i64 %32
%i648B

	full_text
	
i64 %32
Abitcast8B4
2
	full_text%
#
!%34 = bitcast double* %33 to i64*
-double*8B

	full_text

double* %33
Hload8B>
<
	full_text/
-
+%35 = load i64, i64* %34, align 8, !tbaa !8
'i64*8B

	full_text


i64* %34
4add8B+
)
	full_text

%36 = add nsw i32 %2, -1
5mul8B,
*
	full_text

%37 = mul nsw i32 %36, %1
%i328B

	full_text
	
i32 %36
1add8B(
&
	full_text

%38 = add i32 %37, %4
%i328B

	full_text
	
i32 %37
1add8B(
&
	full_text

%39 = add i32 %38, %7
%i328B

	full_text
	
i32 %38
$i328B

	full_text


i32 %7
2add8B)
'
	full_text

%40 = add i32 %39, %14
%i328B

	full_text
	
i32 %39
%i328B

	full_text
	
i32 %14
6sext8B,
*
	full_text

%41 = sext i32 %40 to i64
%i328B

	full_text
	
i32 %40
^getelementptr8BK
I
	full_text<
:
8%42 = getelementptr inbounds double, double* %0, i64 %41
%i648B

	full_text
	
i64 %41
Abitcast8B4
2
	full_text%
#
!%43 = bitcast double* %42 to i64*
-double*8B

	full_text

double* %42
Hstore8B=
;
	full_text.
,
*store i64 %35, i64* %43, align 8, !tbaa !8
%i648B

	full_text
	
i64 %35
'i64*8B

	full_text


i64* %43
'br8B

	full_text

br label %44
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %2
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %1
-; undefined function B

	full_text

 
#i648B

	full_text	

i64 1
$i328B

	full_text


i32 -2
$i328B

	full_text


i32 -1
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1       	
 		                       !    "# "" $% $$ &' &( && )* )) +, ++ -. -- /0 /1 // 22 34 35 33 67 68 66 9: 99 ;< ;; => == ?@ ?? AA BC BB DE DD FG FH FF IJ IK II LM LL NO NN PQ PP RS RT RR UW W W AX X $X 2X DY Y +Y ;Y NZ Z Z Z 2Z B    
	            !  # %$ ' (& *) ,+ ." 0- 12 4 53 7 86 :9 <; >= @A CB ED G HF J KI ML ON Q? SP T  VU V V [[ [[  [[ \ 	] ^ A_ ` "
kernel_comm3_2"
_Z13get_global_idj*?
npb-MG-kernel_comm3_2.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02
 
transfer_bytes_log1p
?`A

wgsize
 

transfer_bytes
??I

wgsize_log1p
?`A

devmap_label
 