
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

br i1 %8, label %9, label %43
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
8trunc8B-
+
	full_text

%11 = trunc i64 %10 to i32
%i648B

	full_text
	
i64 %10
4add8B+
)
	full_text

%12 = add nsw i32 %3, -2
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
5mul8B,
*
	full_text

%15 = mul nsw i32 %11, %1
%i328B

	full_text
	
i32 %11
1add8B(
&
	full_text

%16 = add i32 %14, %4
%i328B

	full_text
	
i32 %14
1add8B(
&
	full_text

%17 = add i32 %16, %7
%i328B

	full_text
	
i32 %16
$i328B

	full_text


i32 %7
2add8B)
'
	full_text

%18 = add i32 %17, %15
%i328B

	full_text
	
i32 %17
%i328B

	full_text
	
i32 %15
6sext8B,
*
	full_text

%19 = sext i32 %18 to i64
%i328B

	full_text
	
i32 %18
^getelementptr8BK
I
	full_text<
:
8%20 = getelementptr inbounds double, double* %0, i64 %19
%i648B

	full_text
	
i64 %19
Abitcast8B4
2
	full_text%
#
!%21 = bitcast double* %20 to i64*
-double*8B

	full_text

double* %20
Hload8B>
<
	full_text/
-
+%22 = load i64, i64* %21, align 8, !tbaa !8
'i64*8B

	full_text


i64* %21
0add8B'
%
	full_text

%23 = add i32 %7, %4
$i328B

	full_text


i32 %7
2add8B)
'
	full_text

%24 = add i32 %23, %15
%i328B

	full_text
	
i32 %23
%i328B

	full_text
	
i32 %15
6sext8B,
*
	full_text

%25 = sext i32 %24 to i64
%i328B

	full_text
	
i32 %24
^getelementptr8BK
I
	full_text<
:
8%26 = getelementptr inbounds double, double* %0, i64 %25
%i648B

	full_text
	
i64 %25
Abitcast8B4
2
	full_text%
#
!%27 = bitcast double* %26 to i64*
-double*8B

	full_text

double* %26
Hstore8B=
;
	full_text.
,
*store i64 %22, i64* %27, align 8, !tbaa !8
%i648B

	full_text
	
i64 %22
'i64*8B

	full_text


i64* %27
1add8B(
&
	full_text

%28 = add i32 %13, %4
%i328B

	full_text
	
i32 %13
1add8B(
&
	full_text

%29 = add i32 %28, %7
%i328B

	full_text
	
i32 %28
$i328B

	full_text


i32 %7
2add8B)
'
	full_text

%30 = add i32 %29, %15
%i328B

	full_text
	
i32 %29
%i328B

	full_text
	
i32 %15
6sext8B,
*
	full_text

%31 = sext i32 %30 to i64
%i328B

	full_text
	
i32 %30
^getelementptr8BK
I
	full_text<
:
8%32 = getelementptr inbounds double, double* %0, i64 %31
%i648B

	full_text
	
i64 %31
Abitcast8B4
2
	full_text%
#
!%33 = bitcast double* %32 to i64*
-double*8B

	full_text

double* %32
Hload8B>
<
	full_text/
-
+%34 = load i64, i64* %33, align 8, !tbaa !8
'i64*8B

	full_text


i64* %33
4add8B+
)
	full_text

%35 = add nsw i32 %3, -1
2mul8B)
'
	full_text

%36 = mul i32 %13, %35
%i328B

	full_text
	
i32 %13
%i328B

	full_text
	
i32 %35
1add8B(
&
	full_text

%37 = add i32 %36, %4
%i328B

	full_text
	
i32 %36
1add8B(
&
	full_text

%38 = add i32 %37, %7
%i328B

	full_text
	
i32 %37
$i328B

	full_text


i32 %7
2add8B)
'
	full_text

%39 = add i32 %38, %15
%i328B

	full_text
	
i32 %38
%i328B

	full_text
	
i32 %15
6sext8B,
*
	full_text

%40 = sext i32 %39 to i64
%i328B

	full_text
	
i32 %39
^getelementptr8BK
I
	full_text<
:
8%41 = getelementptr inbounds double, double* %0, i64 %40
%i648B

	full_text
	
i64 %40
Abitcast8B4
2
	full_text%
#
!%42 = bitcast double* %41 to i64*
-double*8B

	full_text

double* %41
Hstore8B=
;
	full_text.
,
*store i64 %34, i64* %42, align 8, !tbaa !8
%i648B

	full_text
	
i64 %34
'i64*8B

	full_text


i64* %42
'br8B

	full_text

br label %43
$ret8B

	full_text


ret void
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
$i328B

	full_text


i32 %2
$i328B

	full_text


i32 %3
-; undefined function B

	full_text

 
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
 		                       !    "# "" $% $& $$ '( '' )* )) +, ++ -. -/ -- 01 00 23 24 22 56 57 55 89 88 :; :: <= << >? >> @@ AB AC AA DE DD FG FH FF IJ IK II LM LL NO NN PQ PP RS RT RR UW W "W 0W DX X )X :X NY Y Y Z [ [ @    
  	          ! #" % &$ (' *) ,  .+ / 10 3 42 6 75 98 ;: =< ? B@ CA ED G HF J KI ML ON Q> SP T  VU V V \\ \\  \\ ] ^ @_ ` "
kernel_comm3_3"
_Z13get_global_idj*?
npb-MG-kernel_comm3_3.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

transfer_bytes	
????

devmap_label

 
transfer_bytes_log1p
???A

wgsize
 

wgsize_log1p
???A